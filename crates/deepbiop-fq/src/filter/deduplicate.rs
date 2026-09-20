//! Deduplication of FASTQ records based on sequence identity.

use super::{Filter, FilterWithReason};
use ahash::HashSet;
use derive_builder::Builder;
use noodles::fastq;

/// Deduplicate records based on sequence content.
///
/// Keeps the first occurrence of every distinct sequence (quality scores are
/// ignored). Every distinct sequence is held in memory for the lifetime of the
/// filter, so memory grows with the number of unique reads.
///
/// # Examples
///
/// ```no_run
/// use deepbiop_fq::filter::{Deduplicator, DeduplicatorBuilder, Filter};
/// use noodles::fastq;
///
/// let mut dedup = Deduplicator::new();
///
/// let record1 = fastq::Record::new(
///     fastq::record::Definition::new("read1", ""),
///     b"ACGT".to_vec(),
///     b"IIII".to_vec(),
/// );
///
/// let record2 = fastq::Record::new(
///     fastq::record::Definition::new("read2", ""),
///     b"ACGT".to_vec(), // Same sequence
///     b"!!!!".to_vec(), // Different quality
/// );
///
/// assert!(dedup.passes(&record1)); // First occurrence, passes
/// assert!(!dedup.passes(&record2)); // Duplicate sequence, fails
/// ```
#[derive(Debug, Clone, Default, Builder)]
#[builder(setter(into), default)]
pub struct Deduplicator {
    /// Distinct sequences seen so far
    #[builder(setter(skip))]
    seen_sequences: HashSet<Vec<u8>>,
}

impl Deduplicator {
    /// Create a new deduplicator.
    ///
    /// By default, keeps the first occurrence of each unique sequence.
    pub fn new() -> Self {
        Self::default()
    }

    /// Check if a sequence has been seen before.
    pub fn is_duplicate(&self, sequence: &[u8]) -> bool {
        self.seen_sequences.contains(sequence)
    }

    /// Get the number of unique sequences seen so far.
    pub fn unique_count(&self) -> usize {
        self.seen_sequences.len()
    }

    /// Clear all tracked sequences (reset the deduplicator).
    pub fn clear(&mut self) {
        self.seen_sequences.clear();
    }
}

impl Filter for Deduplicator {
    fn passes(&mut self, record: &fastq::Record) -> bool {
        if self.seen_sequences.contains(record.sequence()) {
            return false;
        }
        self.seen_sequences.insert(record.sequence().to_vec());
        true
    }
}

impl FilterWithReason for Deduplicator {
    fn check(&mut self, record: &fastq::Record) -> Option<String> {
        if self.passes(record) {
            None
        } else {
            Some(format!(
                "Duplicate sequence (length: {})",
                record.sequence().len()
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_record(name: &str, seq: &[u8]) -> fastq::Record {
        let qual = vec![b'I'; seq.len()];
        fastq::Record::new(fastq::record::Definition::new(name, ""), seq.to_vec(), qual)
    }

    #[test]
    fn test_no_duplicates() {
        let mut dedup = Deduplicator::new();

        assert!(dedup.passes(&create_record("read1", b"ACGT")));
        assert!(dedup.passes(&create_record("read2", b"TTGG")));
        assert!(dedup.passes(&create_record("read3", b"AAAA")));

        assert_eq!(dedup.unique_count(), 3);
    }

    #[test]
    fn test_exact_duplicates() {
        let mut dedup = Deduplicator::new();

        // First occurrence passes
        assert!(dedup.passes(&create_record("read1", b"ACGT")));
        assert_eq!(dedup.unique_count(), 1);

        // Duplicate fails
        assert!(!dedup.passes(&create_record("read2", b"ACGT")));
        assert_eq!(dedup.unique_count(), 1);

        // Another duplicate fails
        assert!(!dedup.passes(&create_record("read3", b"ACGT")));
        assert_eq!(dedup.unique_count(), 1);

        // Different sequence passes
        assert!(dedup.passes(&create_record("read4", b"TTGG")));
        assert_eq!(dedup.unique_count(), 2);
    }

    #[test]
    fn test_quality_ignored() {
        let mut dedup = Deduplicator::new();

        // Same sequence, different quality
        let record1 = fastq::Record::new(
            fastq::record::Definition::new("read1", ""),
            b"ACGT".to_vec(),
            b"IIII".to_vec(),
        );

        let record2 = fastq::Record::new(
            fastq::record::Definition::new("read2", ""),
            b"ACGT".to_vec(),
            b"!!!!".to_vec(), // Different quality
        );

        assert!(dedup.passes(&record1)); // First passes
        assert!(!dedup.passes(&record2)); // Duplicate fails
    }

    #[test]
    fn test_is_duplicate() {
        let mut dedup = Deduplicator::new();

        assert!(!dedup.is_duplicate(b"ACGT"));

        dedup.passes(&create_record("read1", b"ACGT"));

        assert!(dedup.is_duplicate(b"ACGT"));
        assert!(!dedup.is_duplicate(b"TTGG"));
    }

    #[test]
    fn test_clear() {
        let mut dedup = Deduplicator::new();

        dedup.passes(&create_record("read1", b"ACGT"));
        dedup.passes(&create_record("read2", b"TTGG"));
        assert_eq!(dedup.unique_count(), 2);

        dedup.clear();
        assert_eq!(dedup.unique_count(), 0);

        // After clearing, same sequences can pass again
        assert!(dedup.passes(&create_record("read3", b"ACGT")));
        assert_eq!(dedup.unique_count(), 1);
    }

    #[test]
    fn test_filter_with_reason() {
        let mut dedup = Deduplicator::new();

        let record1 = create_record("read1", b"ACGT");
        assert_eq!(dedup.check(&record1), None); // First occurrence

        let record2 = create_record("read2", b"ACGT");
        assert_eq!(
            dedup.check(&record2),
            Some("Duplicate sequence (length: 4)".to_string())
        );
    }

    #[test]
    fn test_builder() {
        let mut dedup = DeduplicatorBuilder::default().build().unwrap();

        assert!(dedup.passes(&create_record("read1", b"ACGT")));
        assert!(!dedup.passes(&create_record("read2", b"ACGT")));
    }
}
