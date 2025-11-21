//! Length-based filtering for FASTQ records.

use super::{Filter, FilterWithReason};
use derive_builder::Builder;
use noodles::fastq;

/// Filter records based on sequence length.
///
/// # Examples
///
/// ```no_run
/// use deepbiop_fq::filter::{LengthFilter, LengthFilterBuilder, Filter};
/// use noodles::fastq;
///
/// // Filter sequences between 50 and 500 bp
/// let mut filter = LengthFilterBuilder::default()
///     .min_length(Some(50))
///     .max_length(Some(500))
///     .build()
///     .unwrap();
///
/// let record = fastq::Record::new(
///     fastq::record::Definition::new("read1", ""),
///     b"ACGTACGTACGT".to_vec(),
///     b"IIIIIIIIIIII".to_vec(),
/// );
///
/// assert!(filter.passes(&record)); // 12 bp, within range
/// ```
#[derive(Debug, Clone, Builder)]
#[builder(setter(into), default)]
#[derive(Default)]
pub struct LengthFilter {
    /// Minimum sequence length (inclusive). `None` means no minimum.
    min_length: Option<usize>,

    /// Maximum sequence length (inclusive). `None` means no maximum.
    max_length: Option<usize>,
}

impl LengthFilter {
    /// Create a new length filter with optional min/max bounds.
    ///
    /// # Arguments
    ///
    /// * `min_length` - Minimum sequence length (inclusive), or `None` for no minimum
    /// * `max_length` - Maximum sequence length (inclusive), or `None` for no maximum
    pub fn new(min_length: Option<usize>, max_length: Option<usize>) -> Self {
        Self {
            min_length,
            max_length,
        }
    }

    /// Create a filter that only accepts sequences with at least `min` bases.
    pub fn min_only(min: usize) -> Self {
        Self {
            min_length: Some(min),
            max_length: None,
        }
    }

    /// Create a filter that only accepts sequences with at most `max` bases.
    pub fn max_only(max: usize) -> Self {
        Self {
            min_length: None,
            max_length: Some(max),
        }
    }

    /// Create a filter that accepts sequences within a specific length range.
    pub fn range(min: usize, max: usize) -> Self {
        Self {
            min_length: Some(min),
            max_length: Some(max),
        }
    }

    /// Get the minimum length threshold.
    pub fn min_length(&self) -> Option<usize> {
        self.min_length
    }

    /// Get the maximum length threshold.
    pub fn max_length(&self) -> Option<usize> {
        self.max_length
    }
}

impl Filter for LengthFilter {
    fn passes(&mut self, record: &fastq::Record) -> bool {
        let length = record.sequence().len();

        if let Some(min) = self.min_length {
            if length < min {
                return false;
            }
        }

        if let Some(max) = self.max_length {
            if length > max {
                return false;
            }
        }

        true
    }
}

impl FilterWithReason for LengthFilter {
    fn check(&mut self, record: &fastq::Record) -> Option<String> {
        let length = record.sequence().len();

        if let Some(min) = self.min_length {
            if length < min {
                return Some(format!("Sequence too short: {} < {} (min)", length, min));
            }
        }

        if let Some(max) = self.max_length {
            if length > max {
                return Some(format!("Sequence too long: {} > {} (max)", length, max));
            }
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_record(seq: &[u8]) -> fastq::Record {
        let qual = vec![b'I'; seq.len()]; // High quality scores
        fastq::Record::new(
            fastq::record::Definition::new("test", ""),
            seq.to_vec(),
            qual,
        )
    }

    #[test]
    fn test_no_filter() {
        let mut filter = LengthFilter::new(None, None);

        assert!(filter.passes(&create_record(b"A")));
        assert!(filter.passes(&create_record(b"ACGT")));
        assert!(filter.passes(&create_record(&vec![b'A'; 1000])));
    }

    #[test]
    fn test_min_length_only() {
        let mut filter = LengthFilter::min_only(50);

        assert!(!filter.passes(&create_record(b"ACGT"))); // 4 bp < 50
        assert!(!filter.passes(&create_record(&[b'A'; 49]))); // 49 bp < 50
        assert!(filter.passes(&create_record(&[b'A'; 50]))); // 50 bp == 50
        assert!(filter.passes(&create_record(&[b'A'; 100]))); // 100 bp > 50
    }

    #[test]
    fn test_max_length_only() {
        let mut filter = LengthFilter::max_only(500);

        assert!(filter.passes(&create_record(b"ACGT"))); // 4 bp < 500
        assert!(filter.passes(&create_record(&vec![b'A'; 500]))); // 500 bp == 500
        assert!(!filter.passes(&create_record(&vec![b'A'; 501]))); // 501 bp > 500
        assert!(!filter.passes(&create_record(&vec![b'A'; 1000]))); // 1000 bp > 500
    }

    #[test]
    fn test_length_range() {
        let mut filter = LengthFilter::range(50, 500);

        assert!(!filter.passes(&create_record(b"ACGT"))); // 4 bp < 50
        assert!(!filter.passes(&create_record(&[b'A'; 49]))); // 49 bp < 50
        assert!(filter.passes(&create_record(&[b'A'; 50]))); // 50 bp == 50
        assert!(filter.passes(&create_record(&vec![b'A'; 250]))); // 250 bp in range
        assert!(filter.passes(&create_record(&vec![b'A'; 500]))); // 500 bp == 500
        assert!(!filter.passes(&create_record(&vec![b'A'; 501]))); // 501 bp > 500
    }

    #[test]
    fn test_filter_with_reason() {
        let mut filter = LengthFilter::range(50, 500);

        let short_record = create_record(b"ACGT"); // 4 bp
        assert_eq!(
            filter.check(&short_record),
            Some("Sequence too short: 4 < 50 (min)".to_string())
        );

        let long_record = create_record(&vec![b'A'; 600]); // 600 bp
        assert_eq!(
            filter.check(&long_record),
            Some("Sequence too long: 600 > 500 (max)".to_string())
        );

        let good_record = create_record(&[b'A'; 100]); // 100 bp
        assert_eq!(filter.check(&good_record), None);
    }

    #[test]
    fn test_builder() {
        let mut filter = LengthFilterBuilder::default()
            .min_length(Some(50))
            .max_length(Some(500))
            .build()
            .unwrap();

        assert!(!filter.passes(&create_record(b"ACGT")));
        assert!(filter.passes(&create_record(&[b'A'; 100])));
        assert!(!filter.passes(&create_record(&vec![b'A'; 600])));
    }

    #[test]
    fn test_getters() {
        let filter = LengthFilter::range(50, 500);
        assert_eq!(filter.min_length(), Some(50));
        assert_eq!(filter.max_length(), Some(500));

        let filter_no_bounds = LengthFilter::new(None, None);
        assert_eq!(filter_no_bounds.min_length(), None);
        assert_eq!(filter_no_bounds.max_length(), None);
    }
}
