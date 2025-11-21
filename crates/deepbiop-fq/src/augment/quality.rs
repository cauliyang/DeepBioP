//! Quality score simulation for FASTQ records.

use rand::Rng;
use rand_distr::{Distribution, Normal};

/// Quality score distribution model.
#[derive(Debug, Clone)]
pub enum QualityModel {
    /// Uniform distribution between min and max quality scores.
    Uniform { min: u8, max: u8 },

    /// Normal distribution with mean and standard deviation.
    /// Clipped to valid Phred score range [0, 93].
    Normal { mean: f64, std_dev: f64 },

    /// High quality (typical modern Illumina): mean ~37, std ~2
    HighQuality,

    /// Medium quality (typical older platforms): mean ~28, std ~5
    MediumQuality,

    /// Degrading quality (typical read end): decreasing quality along read
    Degrading {
        start_mean: f64,
        end_mean: f64,
        std_dev: f64,
    },
}

/// Quality score simulator for FASTQ records.
///
/// Simulates realistic Phred quality scores for sequences, useful for:
/// - Converting FASTA to FASTQ
/// - Data augmentation
/// - Benchmarking quality-aware algorithms
///
/// Quality scores are encoded as Phred+33 (Sanger/Illumina 1.8+).
///
/// # Examples
///
/// ```
/// use deepbiop_fq::augment::quality::{QualitySimulator, QualityModel};
///
/// // High quality simulation (typical modern Illumina)
/// let mut sim = QualitySimulator::new(QualityModel::HighQuality, Some(42));
/// let quality = sim.generate(100); // 100 bp read
/// assert_eq!(quality.len(), 100);
///
/// // Degrading quality (typical for longer reads)
/// let mut sim2 = QualitySimulator::new(
///     QualityModel::Degrading {
///         start_mean: 40.0,
///         end_mean: 25.0,
///         std_dev: 3.0,
///     },
///     None
/// );
/// let quality = sim2.generate(150);
/// ```
#[derive(Debug, Clone)]
pub struct QualitySimulator {
    model: QualityModel,
    rng: rand::rngs::StdRng,
}

impl QualitySimulator {
    /// Create a new quality score simulator.
    ///
    /// # Arguments
    ///
    /// * `model` - Quality score distribution model
    /// * `seed` - Optional RNG seed for reproducibility
    pub fn new(model: QualityModel, seed: Option<u64>) -> Self {
        use rand::SeedableRng;

        let rng = match seed {
            Some(seed) => rand::rngs::StdRng::seed_from_u64(seed),
            None => {
                // Create a seeded RNG from thread_rng
                let seed = rand::random::<u64>();
                rand::rngs::StdRng::seed_from_u64(seed)
            }
        };

        Self { model, rng }
    }

    /// Generate quality scores for a sequence of given length.
    ///
    /// Returns a vector of ASCII quality scores (Phred+33 encoding).
    ///
    /// # Arguments
    ///
    /// * `length` - Number of quality scores to generate
    ///
    /// # Returns
    ///
    /// Vector of ASCII quality scores (! to ~, representing Phred 0-93)
    pub fn generate(&mut self, length: usize) -> Vec<u8> {
        let mut result = Vec::with_capacity(length);

        match &self.model {
            QualityModel::Uniform { min, max } => {
                let min_q = *min;
                let max_q = *max;
                for _ in 0..length {
                    let q = self.rng.random_range(min_q..=max_q);
                    result.push(self.phred_to_ascii(q));
                }
            }

            QualityModel::Normal { mean, std_dev } => {
                let normal = Normal::new(*mean, *std_dev).unwrap();
                for _ in 0..length {
                    let q: f64 = normal.sample(&mut self.rng);
                    let q_clipped = q.clamp(0.0, 93.0) as u8;
                    result.push(self.phred_to_ascii(q_clipped));
                }
            }

            QualityModel::HighQuality => {
                // Modern Illumina: mean ~37, std ~2
                let normal = Normal::new(37.0, 2.0).unwrap();
                for _ in 0..length {
                    let q: f64 = normal.sample(&mut self.rng);
                    let q_clipped = q.clamp(0.0, 93.0) as u8;
                    result.push(self.phred_to_ascii(q_clipped));
                }
            }

            QualityModel::MediumQuality => {
                // Older platforms: mean ~28, std ~5
                let normal = Normal::new(28.0, 5.0).unwrap();
                for _ in 0..length {
                    let q: f64 = normal.sample(&mut self.rng);
                    let q_clipped = q.clamp(0.0, 93.0) as u8;
                    result.push(self.phred_to_ascii(q_clipped));
                }
            }

            QualityModel::Degrading {
                start_mean,
                end_mean,
                std_dev,
            } => {
                // Quality decreases linearly along read
                let start = *start_mean;
                let end = *end_mean;
                let std = *std_dev;
                let normal = Normal::new(0.0, std).unwrap();

                for i in 0..length {
                    // Linear interpolation
                    let fraction = i as f64 / length.max(1) as f64;
                    let mean_at_pos = start + (end - start) * fraction;
                    let noise: f64 = normal.sample(&mut self.rng);
                    let q = mean_at_pos + noise;
                    let q_clipped = q.clamp(0.0, 93.0) as u8;
                    result.push(self.phred_to_ascii(q_clipped));
                }
            }
        }

        result
    }

    /// Convert Phred quality score to ASCII character (Phred+33).
    #[inline]
    fn phred_to_ascii(&self, phred: u8) -> u8 {
        phred.min(93) + 33
    }

    /// Convert ASCII character to Phred quality score.
    #[inline]
    pub fn ascii_to_phred(ascii: u8) -> u8 {
        ascii.saturating_sub(33)
    }
}

/// Builder for QualitySimulator with convenient presets.
pub struct QualitySimulatorBuilder {
    model: Option<QualityModel>,
    seed: Option<u64>,
}

impl QualitySimulatorBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self {
            model: None,
            seed: None,
        }
    }

    /// Set the quality model.
    pub fn model(mut self, model: QualityModel) -> Self {
        self.model = Some(model);
        self
    }

    /// Use high quality preset (modern Illumina).
    pub fn high_quality(mut self) -> Self {
        self.model = Some(QualityModel::HighQuality);
        self
    }

    /// Use medium quality preset.
    pub fn medium_quality(mut self) -> Self {
        self.model = Some(QualityModel::MediumQuality);
        self
    }

    /// Use degrading quality model.
    pub fn degrading(mut self, start_mean: f64, end_mean: f64, std_dev: f64) -> Self {
        self.model = Some(QualityModel::Degrading {
            start_mean,
            end_mean,
            std_dev,
        });
        self
    }

    /// Set random seed for reproducibility.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Build the simulator.
    pub fn build(self) -> QualitySimulator {
        let model = self.model.unwrap_or(QualityModel::HighQuality);
        QualitySimulator::new(model, self.seed)
    }
}

impl Default for QualitySimulatorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uniform_quality() {
        let mut sim = QualitySimulator::new(QualityModel::Uniform { min: 20, max: 40 }, Some(42));

        let quality = sim.generate(100);
        assert_eq!(quality.len(), 100);

        // All scores should be in valid range
        for &q_ascii in &quality {
            let q = QualitySimulator::ascii_to_phred(q_ascii);
            assert!((20..=40).contains(&q), "Quality {} out of range", q);
        }
    }

    #[test]
    fn test_high_quality() {
        let mut sim = QualitySimulator::new(QualityModel::HighQuality, Some(42));
        let quality = sim.generate(100);

        // Calculate mean quality
        let mean: f64 = quality
            .iter()
            .map(|&q| QualitySimulator::ascii_to_phred(q) as f64)
            .sum::<f64>()
            / quality.len() as f64;

        // Should be around 37 ± 5
        assert!(
            mean > 32.0 && mean < 42.0,
            "Mean quality {} not near 37",
            mean
        );
    }

    #[test]
    fn test_degrading_quality() {
        let mut sim = QualitySimulator::new(
            QualityModel::Degrading {
                start_mean: 40.0,
                end_mean: 25.0,
                std_dev: 2.0,
            },
            Some(42),
        );

        let quality = sim.generate(150);

        // First 10% should be higher quality than last 10%
        let first_mean: f64 = quality[0..15]
            .iter()
            .map(|&q| QualitySimulator::ascii_to_phred(q) as f64)
            .sum::<f64>()
            / 15.0;

        let last_mean: f64 = quality[135..150]
            .iter()
            .map(|&q| QualitySimulator::ascii_to_phred(q) as f64)
            .sum::<f64>()
            / 15.0;

        assert!(
            first_mean > last_mean,
            "Quality should degrade: {} > {}",
            first_mean,
            last_mean
        );
    }

    #[test]
    fn test_builder() {
        let sim = QualitySimulatorBuilder::new()
            .high_quality()
            .seed(42)
            .build();

        assert!(matches!(sim.model, QualityModel::HighQuality));
    }

    #[test]
    fn test_phred_conversion() {
        let sim = QualitySimulator::new(QualityModel::HighQuality, None);

        // Phred 0 -> ASCII 33 (!)
        assert_eq!(sim.phred_to_ascii(0), 33);

        // Phred 30 -> ASCII 63 (?)
        assert_eq!(sim.phred_to_ascii(30), 63);

        // Phred 40 -> ASCII 73 (I)
        assert_eq!(sim.phred_to_ascii(40), 73);

        // Round trip
        let ascii = sim.phred_to_ascii(35);
        assert_eq!(QualitySimulator::ascii_to_phred(ascii), 35);
    }

    #[test]
    fn test_reproducibility() {
        let mut sim1 = QualitySimulator::new(QualityModel::HighQuality, Some(12345));
        let mut sim2 = QualitySimulator::new(QualityModel::HighQuality, Some(12345));

        let q1 = sim1.generate(50);
        let q2 = sim2.generate(50);

        assert_eq!(q1, q2, "Same seed should produce same output");
    }
}
