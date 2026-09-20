//! Parallel processing utilities for DeepBioP
//!
//! This module provides helper functions for managing parallel execution,
//! including thread pool configuration and worker count calculation.

use std::num::NonZeroUsize;
use std::thread;

/// Calculates the optimal worker count for parallel operations
///
/// This function takes an optional thread count and ensures it doesn't exceed
/// the available parallelism on the system. It's used to standardize thread
/// pool creation across all crates.
///
/// # Arguments
///
/// * `threads` - Optional number of threads to use. `None` uses all available cores; 0 is treated as 1.
///
/// # Returns
///
/// Returns a `NonZeroUsize` representing the calculated worker count, which is
/// the minimum of the requested threads and available system parallelism.
/// Always returns at least 1.
///
/// # Examples
///
/// ```
/// use deepbiop_utils::parallel::calculate_worker_count;
///
/// // Request 4 threads
/// let worker_count = calculate_worker_count(Some(4));
/// assert!(worker_count.get() <= 4);
/// assert!(worker_count.get() >= 1);
///
/// // `None` uses every available core
/// let worker_count = calculate_worker_count(None);
/// assert!(worker_count.get() >= 1);
///
/// // Zero threads defaults to 1
/// let worker_count = calculate_worker_count(Some(0));
/// assert_eq!(worker_count.get(), 1);
/// ```
pub fn calculate_worker_count(threads: Option<usize>) -> NonZeroUsize {
    let available = thread::available_parallelism().unwrap_or(NonZeroUsize::MIN);
    match threads {
        None => available,
        Some(n) => NonZeroUsize::new(n)
            .unwrap_or(NonZeroUsize::MIN)
            .min(available),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_worker_count_default_uses_all_cores() {
        let worker_count = calculate_worker_count(None);
        assert_eq!(
            worker_count,
            thread::available_parallelism().unwrap_or(NonZeroUsize::MIN)
        );
    }

    #[test]
    fn test_calculate_worker_count_with_threads() {
        let worker_count = calculate_worker_count(Some(2));
        assert!(worker_count.get() >= 1);
        assert!(worker_count.get() <= 2);
    }

    #[test]
    fn test_calculate_worker_count_exceeds_available() {
        // Request more threads than available
        let huge_count = 1000;
        let worker_count = calculate_worker_count(Some(huge_count));

        // Should be capped at available parallelism
        let available = thread::available_parallelism().unwrap();
        assert_eq!(worker_count, available);
    }

    #[test]
    fn test_calculate_worker_count_zero_defaults_to_one() {
        // Zero should default to 1
        let worker_count = calculate_worker_count(Some(0));
        assert_eq!(worker_count.get(), 1);
    }

    #[test]
    fn test_calculate_worker_count_respects_system_limit() {
        let available = thread::available_parallelism().unwrap().get();

        // Request exactly the available count
        let worker_count = calculate_worker_count(Some(available));
        assert_eq!(worker_count.get(), available);

        // Request one less than available
        if available > 1 {
            let worker_count = calculate_worker_count(Some(available - 1));
            assert_eq!(worker_count.get(), available - 1);
        }
    }
}
