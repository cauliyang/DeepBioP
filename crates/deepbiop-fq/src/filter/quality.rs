//! Quality score-based filtering for FASTQ records.

use super::{Filter, FilterWithReason};
use derive_builder::Builder;
use noodles::fastq;

/// Filter records based on quality scores.
///
/// # Examples
///
/// ```no_run
/// use deepbiop_fq::filter::{QualityFilter, QualityFilterBuilder, Filter};
/// use noodles::fastq;
///
/// // Filter sequences with mean quality >= 20
/// let mut filter = QualityFilterBuilder::default()
///     .min_mean_quality(Some(20.0))
///     .quality_offset(33) // Phred+33 encoding
///     .build()
///     .unwrap();
///
/// let record = fastq::Record::new(
///     fastq::record::Definition::new("read1", ""),
///     b"ACGTACGT".to_vec(),
///     b"IIIIIIII".to_vec(), // High quality
/// );
///
/// assert!(filter.passes(&record));
/// ```
#[derive(Debug, Clone, Builder)]
#[builder(setter(into), default)]
pub struct QualityFilter {
    /// Minimum mean quality score. `None` means no minimum.
    min_mean_quality: Option<f64>,

    /// Minimum quality score for any single base. `None` means no minimum.
    min_base_quality: Option<u8>,

    /// Quality score offset (33 for Phred+33, 64 for Phred+64).
    #[builder(default = "33")]
    quality_offset: u8,
}

impl Default for QualityFilter {
    fn default() -> Self {
        Self {
            min_mean_quality: None,
            min_base_quality: None,
            quality_offset: 33, // Phred+33 (Sanger/Illumina 1.8+)
        }
    }
}

impl QualityFilter {
    /// Create a new quality filter.
    ///
    /// # Arguments
    ///
    /// * `min_mean_quality` - Minimum mean quality score, or `None` for no minimum
    /// * `min_base_quality` - Minimum quality for any single base, or `None` for no minimum
    /// * `quality_offset` - Quality score encoding offset (typically 33)
    pub fn new(
        min_mean_quality: Option<f64>,
        min_base_quality: Option<u8>,
        quality_offset: u8,
    ) -> Self {
        Self {
            min_mean_quality,
            min_base_quality,
            quality_offset,
        }
    }

    /// Create a filter based on mean quality only.
    pub fn mean_quality(min_mean: f64, quality_offset: u8) -> Self {
        Self {
            min_mean_quality: Some(min_mean),
            min_base_quality: None,
            quality_offset,
        }
    }

    /// Create a filter based on minimum base quality only.
    pub fn base_quality(min_base: u8, quality_offset: u8) -> Self {
        Self {
            min_mean_quality: None,
            min_base_quality: Some(min_base),
            quality_offset,
        }
    }

    /// Calculate mean quality score for a record.
    pub fn calculate_mean_quality(&self, record: &fastq::Record) -> f64 {
        let quality_scores = record.quality_scores();
        if quality_scores.is_empty() {
            return 0.0;
        }

        let sum: u64 = quality_scores
            .iter()
            .map(|&q| (q.saturating_sub(self.quality_offset)) as u64)
            .sum();

        sum as f64 / quality_scores.len() as f64
    }

    /// Get the minimum base quality score for a record.
    pub fn calculate_min_base_quality(&self, record: &fastq::Record) -> u8 {
        record
            .quality_scores()
            .iter()
            .map(|&q| q.saturating_sub(self.quality_offset))
            .min()
            .unwrap_or(0)
    }

    /// Get the quality offset.
    pub fn quality_offset(&self) -> u8 {
        self.quality_offset
    }

    /// Get the minimum mean quality threshold.
    pub fn min_mean_quality(&self) -> Option<f64> {
        self.min_mean_quality
    }

    /// Get the minimum base quality threshold.
    pub fn min_base_quality(&self) -> Option<u8> {
        self.min_base_quality
    }
}

impl Filter for QualityFilter {
    fn passes(&mut self, record: &fastq::Record) -> bool {
        // Check mean quality
        if let Some(min_mean) = self.min_mean_quality {
            let mean_quality = self.calculate_mean_quality(record);
            if mean_quality < min_mean {
                return false;
            }
        }

        // Check minimum base quality
        if let Some(min_base) = self.min_base_quality {
            let min_quality = self.calculate_min_base_quality(record);
            if min_quality < min_base {
                return false;
            }
        }

        true
    }
}

impl FilterWithReason for QualityFilter {
    fn check(&mut self, record: &fastq::Record) -> Option<String> {
        // Check mean quality
        if let Some(min_mean) = self.min_mean_quality {
            let mean_quality = self.calculate_mean_quality(record);
            if mean_quality < min_mean {
                return Some(format!(
                    "Mean quality too low: {:.2} < {:.2}",
                    mean_quality, min_mean
                ));
            }
        }

        // Check minimum base quality
        if let Some(min_base) = self.min_base_quality {
            let min_quality = self.calculate_min_base_quality(record);
            if min_quality < min_base {
                return Some(format!(
                    "Minimum base quality too low: {} < {}",
                    min_quality, min_base
                ));
            }
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_record(qual_chars: &[u8]) -> fastq::Record {
        let seq = vec![b'A'; qual_chars.len()];
        fastq::Record::new(
            fastq::record::Definition::new("test", ""),
            seq,
            qual_chars.to_vec(),
        )
    }

    #[test]
    fn test_calculate_mean_quality() {
        let filter = QualityFilter::default(); // Phred+33

        // Quality "IIII" = [73, 73, 73, 73] - 33 = [40, 40, 40, 40]
        let record = create_record(b"IIII");
        assert_eq!(filter.calculate_mean_quality(&record), 40.0);

        // Quality "!!!!" = [33, 33, 33, 33] - 33 = [0, 0, 0, 0]
        let record = create_record(b"!!!!");
        assert_eq!(filter.calculate_mean_quality(&record), 0.0);

        // Quality "5555" = [53, 53, 53, 53] - 33 = [20, 20, 20, 20]
        let record = create_record(b"5555");
        assert_eq!(filter.calculate_mean_quality(&record), 20.0);
    }

    #[test]
    fn test_calculate_min_base_quality() {
        let filter = QualityFilter::default(); // Phred+33

        // Quality "I!!I" = [73, 33, 33, 73] - 33 = [40, 0, 0, 40]
        let record = create_record(b"I!!I");
        assert_eq!(filter.calculate_min_base_quality(&record), 0);

        // Quality "5555" = [53, 53, 53, 53] - 33 = [20, 20, 20, 20]
        let record = create_record(b"5555");
        assert_eq!(filter.calculate_min_base_quality(&record), 20);
    }

    #[test]
    fn test_mean_quality_filter() {
        let mut filter = QualityFilter::mean_quality(20.0, 33);

        // Mean quality 40 (passes)
        assert!(filter.passes(&create_record(b"IIII")));

        // Mean quality 20 (passes, exactly at threshold)
        assert!(filter.passes(&create_record(b"5555")));

        // Mean quality 0 (fails)
        assert!(!filter.passes(&create_record(b"!!!!")));

        // Mixed quality: "II!!" = [40, 40, 0, 0], mean = 20 (passes)
        assert!(filter.passes(&create_record(b"II!!")));

        // Mixed quality: "I!!!" = [40, 0, 0, 0], mean = 10 (fails)
        assert!(!filter.passes(&create_record(b"I!!!")));
    }

    #[test]
    fn test_base_quality_filter() {
        let mut filter = QualityFilter::base_quality(20, 33);

        // All bases >= 20 (passes)
        assert!(filter.passes(&create_record(b"5555")));
        assert!(filter.passes(&create_record(b"IIII")));

        // Has base < 20 (fails)
        assert!(!filter.passes(&create_record(b"!!!!")));
        assert!(!filter.passes(&create_record(b"I!!I"))); // Min is 0
        assert!(!filter.passes(&create_record(b"5!!5"))); // Min is 0
    }

    #[test]
    fn test_combined_filters() {
        let mut filter = QualityFilter::new(Some(20.0), Some(15), 33);

        // Mean 40, min 40 (passes both)
        assert!(filter.passes(&create_record(b"IIII")));

        // Mean 20, min 20 (passes both)
        assert!(filter.passes(&create_record(b"5555")));

        // Mean 0, min 0 (fails both)
        assert!(!filter.passes(&create_record(b"!!!!")));

        // Mean 20, min 0 (passes mean, fails base)
        // "II!!" = [40, 40, 0, 0], mean = 20, min = 0
        assert!(!filter.passes(&create_record(b"II!!")));
    }

    #[test]
    fn test_filter_with_reason() {
        let mut filter = QualityFilter::new(Some(25.0), Some(20), 33);

        // Good quality
        let good_record = create_record(b"IIII"); // Mean 40, min 40
        assert_eq!(filter.check(&good_record), None);

        // Low mean quality
        let low_mean_record = create_record(b"5555"); // Mean 20, min 20
        assert_eq!(
            filter.check(&low_mean_record),
            Some("Mean quality too low: 20.00 < 25.00".to_string())
        );

        // Low base quality
        let low_base_record = create_record(b"I!!I"); // Mean 20, min 0
        assert_eq!(
            filter.check(&low_base_record),
            Some("Mean quality too low: 20.00 < 25.00".to_string())
        );
    }

    #[test]
    fn test_builder() {
        let mut filter = QualityFilterBuilder::default()
            .min_mean_quality(Some(20.0))
            .min_base_quality(Some(15))
            .quality_offset(33)
            .build()
            .unwrap();

        assert!(filter.passes(&create_record(b"IIII")));
        assert!(!filter.passes(&create_record(b"!!!!")));
    }

    #[test]
    fn test_getters() {
        let filter = QualityFilter::new(Some(25.0), Some(20), 33);
        assert_eq!(filter.min_mean_quality(), Some(25.0));
        assert_eq!(filter.min_base_quality(), Some(20));
        assert_eq!(filter.quality_offset(), 33);
    }
}
