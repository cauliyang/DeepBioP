//! Random sampling utilities for DeepBioP
//!
//! This module provides algorithms for random selection from streams of data,
//! particularly useful for sampling from large biological datasets without
//! loading all data into memory.

use rand::{rng, Rng};

/// Performs reservoir sampling to randomly select k items from an iterator
///
/// Reservoir sampling is an algorithm for randomly selecting k items from a stream
/// of unknown size with uniform probability, using only O(k) memory. This is particularly
/// useful for biological data where file sizes can be very large.
///
/// # Algorithm
///
/// The algorithm works in two phases:
/// 1. Fill the reservoir with the first k elements
/// 2. For each subsequent element at index i, randomly decide whether to include it
///    by generating a random number j in [0, i]. If j < k, replace reservoir[j] with
///    the new element.
///
/// This ensures each element has exactly k/n probability of being selected, where n
/// is the total number of elements.
///
/// # Type Parameters
///
/// * `T` - The type of items being sampled
/// * `I` - Iterator type that yields items of type T
///
/// # Arguments
///
/// * `iter` - An iterator over the items to sample from
/// * `k` - The number of items to randomly select
///
/// # Returns
///
/// Returns a `Vec<T>` containing up to k randomly selected items. If the iterator
/// contains fewer than k items, all items are returned.
///
/// # Examples
///
/// ```
/// use deepbiop_utils::sampling::reservoir_sampling;
///
/// // Sample 3 numbers from 1-10
/// let numbers = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
/// let sample = reservoir_sampling(numbers.into_iter(), 3);
/// assert_eq!(sample.len(), 3);
///
/// // Sample from small iterator (fewer items than k)
/// let small = vec![1, 2];
/// let sample = reservoir_sampling(small.into_iter(), 5);
/// assert_eq!(sample.len(), 2);
/// ```
///
/// # Performance
///
/// - Time complexity: O(n) where n is the number of elements in the iterator
/// - Space complexity: O(k) where k is the sample size
pub fn reservoir_sampling<T, I>(iter: I, k: usize) -> Vec<T>
where
    I: Iterator<Item = T>,
{
    let mut rng = rng();
    let mut reservoir = Vec::with_capacity(k);
    let mut count = 0;

    let mut iter = iter.peekable();

    // Phase 1: Fill reservoir with first k elements
    while reservoir.len() < k && iter.peek().is_some() {
        if let Some(item) = iter.next() {
            reservoir.push(item);
            count += 1;
        }
    }

    // Phase 2: Randomly replace elements with decreasing probability
    for item in iter {
        count += 1;
        let j = rng.random_range(0..count);
        if j < k {
            reservoir[j] = item;
        }
    }

    // If we had fewer than k items total, truncate reservoir
    if count < k {
        reservoir.truncate(count);
    }

    reservoir
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn test_reservoir_sampling_exact_size() {
        let data = vec![1, 2, 3, 4, 5];
        let sample = reservoir_sampling(data.into_iter(), 5);
        assert_eq!(sample.len(), 5);

        // All elements should be present (order may vary)
        let sample_set: HashSet<_> = sample.into_iter().collect();
        assert_eq!(sample_set.len(), 5);
    }

    #[test]
    fn test_reservoir_sampling_smaller_than_data() {
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let sample = reservoir_sampling(data.into_iter(), 3);
        assert_eq!(sample.len(), 3);

        // All sampled elements should be unique
        let sample_set: HashSet<_> = sample.into_iter().collect();
        assert_eq!(sample_set.len(), 3);
    }

    #[test]
    fn test_reservoir_sampling_larger_than_data() {
        let data = vec![1, 2, 3];
        let sample = reservoir_sampling(data.clone().into_iter(), 10);

        // Should return all available items
        assert_eq!(sample.len(), 3);

        // Should contain all original items
        let sample_set: HashSet<_> = sample.into_iter().collect();
        let data_set: HashSet<_> = data.into_iter().collect();
        assert_eq!(sample_set, data_set);
    }

    #[test]
    fn test_reservoir_sampling_empty_iterator() {
        let data: Vec<i32> = vec![];
        let sample = reservoir_sampling(data.into_iter(), 5);
        assert_eq!(sample.len(), 0);
    }

    #[test]
    fn test_reservoir_sampling_zero_k() {
        let data = vec![1, 2, 3, 4, 5];
        let sample = reservoir_sampling(data.into_iter(), 0);
        assert_eq!(sample.len(), 0);
    }

    #[test]
    fn test_reservoir_sampling_one_element() {
        let data = vec![42];
        let sample = reservoir_sampling(data.into_iter(), 1);
        assert_eq!(sample.len(), 1);
        assert_eq!(sample[0], 42);
    }

    #[test]
    fn test_reservoir_sampling_distribution() {
        // Statistical test: Each element should appear roughly equally
        // This is a probabilistic test, may occasionally fail due to randomness
        let data: Vec<i32> = (0..100).collect();
        let k = 10;
        let iterations = 1000;

        let mut counts = vec![0; 100];

        for _ in 0..iterations {
            let sample = reservoir_sampling(data.clone().into_iter(), k);
            for &item in &sample {
                counts[item as usize] += 1;
            }
        }

        // Expected count for each element is k * iterations / data.len()
        let expected = k * iterations / data.len();

        // Allow for some variance (within 50% of expected)
        // In a perfect distribution, each would appear exactly 'expected' times
        let min_acceptable = expected / 2;
        let max_acceptable = expected * 3 / 2;

        let in_range = counts
            .iter()
            .filter(|&&c| c >= min_acceptable && c <= max_acceptable)
            .count();

        // At least 80% of elements should fall within acceptable range
        assert!(
            in_range >= 80,
            "Distribution appears biased: {}/{} elements in acceptable range",
            in_range,
            counts.len()
        );
    }

    #[test]
    fn test_reservoir_sampling_with_strings() {
        let data = vec!["apple", "banana", "cherry", "date", "elderberry"];
        let sample = reservoir_sampling(data.into_iter(), 2);
        assert_eq!(sample.len(), 2);
    }
}
