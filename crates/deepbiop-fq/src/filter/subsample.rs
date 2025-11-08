//! Subsampling of FASTQ records.

use super::{Filter, FilterWithReason};
use derive_builder::Builder;
use noodles::fastq;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Subsample records using various strategies.
///
/// Supports:
/// - Random sampling with a given probability/fraction
/// - Every Nth record
/// - First N records
///
/// # Examples
///
/// ```no_run
/// use deepbiop_fq::filter::{Subsampler, SubsamplerBuilder, Filter};
/// use noodles::fastq;
///
/// // Keep 10% of records randomly
/// let mut sampler = SubsamplerBuilder::default()
///     .fraction(Some(0.1))
///     .seed(Some(42))
///     .build()
///     .unwrap();
///
/// let record = fastq::Record::new(
///     fastq::record::Definition::new("read1", ""),
///     b"ACGT".to_vec(),
///     b"IIII".to_vec(),
/// );
///
/// // Will randomly keep ~10% of records
/// let passes = sampler.passes(&record);
/// ```
#[derive(Debug, Clone, Builder)]
#[builder(setter(into), default)]
#[derive(Default)]
pub struct Subsampler {
    /// Fraction of records to keep (0.0 to 1.0). Mutually exclusive with every_nth and first_n.
    fraction: Option<f64>,

    /// Keep every Nth record. Mutually exclusive with fraction and first_n.
    every_nth: Option<usize>,

    /// Keep first N records. Mutually exclusive with fraction and every_nth.
    first_n: Option<usize>,

    /// Random seed for reproducible subsampling (only used with fraction).
    seed: Option<u64>,

    /// Internal counter for tracking record index
    #[builder(setter(skip), default = "0")]
    record_count: usize,

    /// Internal RNG for random subsampling
    #[builder(setter(skip))]
    rng: Option<StdRng>,
}

impl Subsampler {
    /// Create a subsampler that keeps a random fraction of records.
    ///
    /// # Arguments
    ///
    /// * `fraction` - Fraction of records to keep (0.0 to 1.0)
    /// * `seed` - Optional seed for reproducible random sampling
    pub fn random_fraction(fraction: f64, seed: Option<u64>) -> Self {
        assert!(
            (0.0..=1.0).contains(&fraction),
            "Fraction must be between 0.0 and 1.0"
        );

        Self {
            fraction: Some(fraction),
            every_nth: None,
            first_n: None,
            seed,
            record_count: 0,
            rng: None,
        }
    }

    /// Create a subsampler that keeps every Nth record.
    pub fn every_nth(n: usize) -> Self {
        assert!(n > 0, "N must be greater than 0");

        Self {
            fraction: None,
            every_nth: Some(n),
            first_n: None,
            seed: None,
            record_count: 0,
            rng: None,
        }
    }

    /// Create a subsampler that keeps the first N records.
    pub fn first_n(n: usize) -> Self {
        Self {
            fraction: None,
            every_nth: None,
            first_n: Some(n),
            seed: None,
            record_count: 0,
            rng: None,
        }
    }

    /// Get the current record count.
    pub fn record_count(&self) -> usize {
        self.record_count
    }

    /// Reset the internal counter (useful for processing multiple files).
    pub fn reset(&mut self) {
        self.record_count = 0;
        if let Some(seed) = self.seed {
            self.rng = Some(StdRng::seed_from_u64(seed));
        }
    }

    /// Initialize RNG if needed (lazy initialization).
    fn ensure_rng(&mut self) {
        if self.fraction.is_some() && self.rng.is_none() {
            self.rng = Some(if let Some(seed) = self.seed {
                StdRng::seed_from_u64(seed)
            } else {
                StdRng::from_rng(&mut rand::rng())
            });
        }
    }
}

impl Filter for Subsampler {
    fn passes(&mut self, _record: &fastq::Record) -> bool {
        self.record_count += 1;

        if let Some(fraction) = self.fraction {
            // Random sampling
            self.ensure_rng();
            if let Some(ref mut rng) = self.rng {
                rng.random_range(0.0..1.0) < fraction
            } else {
                false
            }
        } else if let Some(n) = self.every_nth {
            // Every Nth record
            (self.record_count - 1).is_multiple_of(n)
        } else if let Some(n) = self.first_n {
            // First N records
            self.record_count <= n
        } else {
            // No sampling strategy specified, keep all
            true
        }
    }
}

impl FilterWithReason for Subsampler {
    fn check(&mut self, record: &fastq::Record) -> Option<String> {
        if self.passes(record) {
            None
        } else if self.fraction.is_some() {
            Some("Filtered by random sampling".to_string())
        } else if self.every_nth.is_some() {
            Some(format!(
                "Filtered by every_nth (keeping every {} record)",
                self.every_nth.unwrap()
            ))
        } else if self.first_n.is_some() {
            Some(format!(
                "Filtered by first_n (kept first {} records)",
                self.first_n.unwrap()
            ))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_record(name: &str) -> fastq::Record {
        fastq::Record::new(
            fastq::record::Definition::new(name, ""),
            b"ACGT".to_vec(),
            b"IIII".to_vec(),
        )
    }

    #[test]
    fn test_every_nth() {
        let mut sampler = Subsampler::every_nth(3);

        // Records: 1, 2, 3, 4, 5, 6, 7, 8, 9
        // Keep:    ✓  ✗  ✗  ✓  ✗  ✗  ✓  ✗  ✗
        assert!(sampler.passes(&create_record("r1"))); // 1st (keep)
        assert!(!sampler.passes(&create_record("r2"))); // 2nd (skip)
        assert!(!sampler.passes(&create_record("r3"))); // 3rd (skip)
        assert!(sampler.passes(&create_record("r4"))); // 4th (keep)
        assert!(!sampler.passes(&create_record("r5"))); // 5th (skip)
        assert!(!sampler.passes(&create_record("r6"))); // 6th (skip)
        assert!(sampler.passes(&create_record("r7"))); // 7th (keep)

        assert_eq!(sampler.record_count(), 7);
    }

    #[test]
    fn test_first_n() {
        let mut sampler = Subsampler::first_n(3);

        assert!(sampler.passes(&create_record("r1"))); // 1st (keep)
        assert!(sampler.passes(&create_record("r2"))); // 2nd (keep)
        assert!(sampler.passes(&create_record("r3"))); // 3rd (keep)
        assert!(!sampler.passes(&create_record("r4"))); // 4th (skip)
        assert!(!sampler.passes(&create_record("r5"))); // 5th (skip)

        assert_eq!(sampler.record_count(), 5);
    }

    #[test]
    fn test_random_fraction_with_seed() {
        let mut sampler1 = Subsampler::random_fraction(0.5, Some(42));
        let mut sampler2 = Subsampler::random_fraction(0.5, Some(42));

        // With same seed, should produce identical results
        for i in 0..100 {
            let record = create_record(&format!("r{}", i));
            let result1 = sampler1.passes(&record);
            let result2 = sampler2.passes(&record);
            assert_eq!(result1, result2);
        }
    }

    #[test]
    fn test_random_fraction_approximately_correct() {
        let mut sampler = Subsampler::random_fraction(0.3, Some(42));

        let mut kept = 0;
        let total = 1000;

        for i in 0..total {
            if sampler.passes(&create_record(&format!("r{}", i))) {
                kept += 1;
            }
        }

        // With 30% sampling and 1000 records, expect ~300 kept
        // Allow 10% deviation (270-330)
        let fraction = kept as f64 / total as f64;
        assert!(
            (0.27..=0.33).contains(&fraction),
            "Expected ~30% sampling, got {:.1}%",
            fraction * 100.0
        );
    }

    #[test]
    fn test_reset() {
        let mut sampler = Subsampler::first_n(3);

        assert!(sampler.passes(&create_record("r1")));
        assert!(sampler.passes(&create_record("r2")));
        assert!(sampler.passes(&create_record("r3")));
        assert!(!sampler.passes(&create_record("r4")));

        sampler.reset();
        assert_eq!(sampler.record_count(), 0);

        // After reset, first 3 should pass again
        assert!(sampler.passes(&create_record("r5")));
        assert!(sampler.passes(&create_record("r6")));
        assert!(sampler.passes(&create_record("r7")));
        assert!(!sampler.passes(&create_record("r8")));
    }

    #[test]
    fn test_no_strategy() {
        let mut sampler = Subsampler::default();

        // With no strategy, all records pass
        for i in 0..10 {
            assert!(sampler.passes(&create_record(&format!("r{}", i))));
        }
    }

    #[test]
    fn test_filter_with_reason() {
        let mut sampler = Subsampler::every_nth(2);

        let record1 = create_record("r1");
        assert_eq!(sampler.check(&record1), None); // 1st passes

        let record2 = create_record("r2");
        assert_eq!(
            sampler.check(&record2),
            Some("Filtered by every_nth (keeping every 2 record)".to_string())
        );
    }

    #[test]
    fn test_builder() {
        let mut sampler = SubsamplerBuilder::default()
            .fraction(Some(0.5))
            .seed(Some(42))
            .build()
            .unwrap();

        // Just verify it compiles and runs
        sampler.passes(&create_record("r1"));
    }

    #[test]
    #[should_panic(expected = "Fraction must be between 0.0 and 1.0")]
    fn test_invalid_fraction() {
        Subsampler::random_fraction(1.5, None);
    }

    #[test]
    #[should_panic(expected = "N must be greater than 0")]
    fn test_invalid_every_nth() {
        Subsampler::every_nth(0);
    }
}
