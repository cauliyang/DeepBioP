//! Deduplication of FASTQ records based on sequence identity.

use super::{Filter, FilterWithReason};
use ahash::HashSet;
use derive_builder::Builder;
use noodles::fastq;

/// Deduplicate records based on sequence content.
///
/// Uses a hash set to track seen sequences. Can deduplicate based on
/// exact sequence match or sequence identity ignoring quality scores.
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
#[derive(Debug, Clone, Builder)]
#[builder(setter(into), default)]
pub struct Deduplicator {
    /// Set of seen sequence hashes
    #[builder(setter(skip))]
    seen_sequences: HashSet<Vec<u8>>,

    /// If true, keep the first occurrence of each unique sequence.
    /// If false, remove all duplicates including the first.
    #[builder(default = "true")]
    keep_first: bool,
}

impl Default for Deduplicator {
    fn default() -> Self {
        Self {
            seen_sequences: HashSet::default(),
            keep_first: true,
        }
    }
}

impl Deduplicator {
    /// Create a new deduplicator.
    ///
    /// By default, keeps the first occurrence of each unique sequence.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a deduplicator that removes all duplicates (including first occurrence).
    pub fn remove_all_duplicates() -> Self {
        Self {
            seen_sequences: HashSet::default(),
            keep_first: false,
        }
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

    /// Get whether the filter keeps first occurrences.
    pub fn keep_first(&self) -> bool {
        self.keep_first
    }
}

impl Filter for Deduplicator {
    fn passes(&mut self, record: &fastq::Record) -> bool {
        let sequence = record.sequence().to_vec();

        if self.seen_sequences.contains(&sequence) {
            // Sequence is a duplicate
            false
        } else {
            // First time seeing this sequence
            self.seen_sequences.insert(sequence);
            self.keep_first
        }
    }
}

impl FilterWithReason for Deduplicator {
    fn check(&mut self, record: &fastq::Record) -> Option<String> {
        let sequence = record.sequence().to_vec();

        if self.seen_sequences.contains(&sequence) {
            Some(format!("Duplicate sequence (length: {})", sequence.len()))
        } else {
            self.seen_sequences.insert(sequence);
            if self.keep_first {
                None
            } else {
                Some("Duplicate sequence (removing all occurrences)".to_string())
            }
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
    fn test_remove_all_duplicates() {
        let mut dedup = Deduplicator::remove_all_duplicates();

        // First occurrence also fails when keep_first = false
        assert!(!dedup.passes(&create_record("read1", b"ACGT")));
        assert_eq!(dedup.unique_count(), 1);

        // Subsequent occurrences also fail
        assert!(!dedup.passes(&create_record("read2", b"ACGT")));
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
        let mut dedup = DeduplicatorBuilder::default()
            .keep_first(true)
            .build()
            .unwrap();

        assert!(dedup.passes(&create_record("read1", b"ACGT")));
        assert!(!dedup.passes(&create_record("read2", b"ACGT")));
    }
}
