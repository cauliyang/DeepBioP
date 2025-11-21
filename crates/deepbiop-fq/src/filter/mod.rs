//! Filtering operations for FASTQ records.
//!
//! This module provides functionality to filter FASTQ records based on various criteria
//! such as sequence length, quality scores, and deduplication.

pub mod deduplicate;
pub mod length;
pub mod quality;
pub mod subsample;

#[cfg(feature = "python")]
pub mod python;

pub use deduplicate::{Deduplicator, DeduplicatorBuilder};
pub use length::{LengthFilter, LengthFilterBuilder};
pub use quality::{QualityFilter, QualityFilterBuilder};
pub use subsample::{Subsampler, SubsamplerBuilder};

use noodles::fastq;

/// Trait for filtering FASTQ records.
pub trait Filter {
    /// Returns `true` if the record passes the filter, `false` otherwise.
    fn passes(&mut self, record: &fastq::Record) -> bool;
}

/// Trait for filtering with detailed information about why a record failed.
pub trait FilterWithReason {
    /// Returns `Some(reason)` if the record fails the filter, `None` if it passes.
    fn check(&mut self, record: &fastq::Record) -> Option<String>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_filter_trait() {
        // Basic test to ensure traits compile
        struct AlwaysPass;
        impl Filter for AlwaysPass {
            fn passes(&mut self, _record: &fastq::Record) -> bool {
                true
            }
        }

        let mut filter = AlwaysPass;
        let record = fastq::Record::new(
            fastq::record::Definition::new("test", ""),
            b"ACGT".to_vec(),
            b"IIII".to_vec(),
        );
        assert!(filter.passes(&record));
    }
}
