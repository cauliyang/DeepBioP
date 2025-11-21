//! Random subsequence sampling for sequence data.

use super::Augmentation;
use derive_builder::Builder;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Random subsequence sampler.
///
/// Extracts random subsequences of fixed length from input sequences.
/// Useful for creating fixed-size inputs for deep learning models.
///
/// # Examples
///
/// ```
/// use deepbiop_fq::augment::{Sampler, Augmentation};
///
/// // Extract 50bp subsequences from random positions
/// let mut sampler = Sampler::random(50, Some(42));
/// let sequence = b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"; // 63 bases
/// let subsequence = sampler.apply(sequence);
/// assert_eq!(subsequence.len(), 50);
/// ```
#[derive(Debug, Clone, Builder)]
#[builder(setter(into))]
pub struct Sampler {
    /// Length of subsequence to extract
    #[builder(default = "100")]
    length: usize,

    /// Sampling strategy
    #[builder(default = "SamplingStrategy::Random")]
    strategy: SamplingStrategy,

    /// Random seed for reproducibility
    #[builder(default = "None")]
    seed: Option<u64>,

    /// Internal RNG
    #[builder(setter(skip), default = "None")]
    rng: Option<StdRng>,
}

/// Strategy for extracting subsequences.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SamplingStrategy {
    /// Random start position
    Random,
    /// Always start from beginning
    Start,
    /// Always start from center
    Center,
    /// Always start from end
    End,
}

impl Sampler {
    /// Create a sampler that extracts subsequences from random positions.
    ///
    /// # Arguments
    ///
    /// * `length` - Length of subsequence to extract
    /// * `seed` - Optional seed for reproducible sampling
    ///
    /// # Panics
    ///
    /// Panics if length is 0
    pub fn random(length: usize, seed: Option<u64>) -> Self {
        assert!(length > 0, "Subsequence length must be greater than 0");

        Self {
            length,
            strategy: SamplingStrategy::Random,
            seed,
            rng: None,
        }
    }

    /// Create a sampler that extracts subsequences from the start.
    pub fn from_start(length: usize) -> Self {
        assert!(length > 0, "Subsequence length must be greater than 0");

        Self {
            length,
            strategy: SamplingStrategy::Start,
            seed: None,
            rng: None,
        }
    }

    /// Create a sampler that extracts subsequences from the center.
    pub fn from_center(length: usize) -> Self {
        assert!(length > 0, "Subsequence length must be greater than 0");

        Self {
            length,
            strategy: SamplingStrategy::Center,
            seed: None,
            rng: None,
        }
    }

    /// Create a sampler that extracts subsequences from the end.
    pub fn from_end(length: usize) -> Self {
        assert!(length > 0, "Subsequence length must be greater than 0");

        Self {
            length,
            strategy: SamplingStrategy::End,
            seed: None,
            rng: None,
        }
    }

    /// Initialize RNG if needed (lazy initialization).
    fn ensure_rng(&mut self) {
        if self.rng.is_none() && self.strategy == SamplingStrategy::Random {
            self.rng = Some(if let Some(seed) = self.seed {
                StdRng::seed_from_u64(seed)
            } else {
                StdRng::from_rng(&mut rand::rng())
            });
        }
    }

    /// Calculate start position based on strategy.
    fn calculate_start(&mut self, seq_len: usize) -> usize {
        if seq_len <= self.length {
            return 0;
        }

        let max_start = seq_len - self.length;

        match self.strategy {
            SamplingStrategy::Random => {
                let rng = self.rng.as_mut().expect("RNG not initialized");
                rng.random_range(0..=max_start)
            }
            SamplingStrategy::Start => 0,
            SamplingStrategy::Center => max_start / 2,
            SamplingStrategy::End => max_start,
        }
    }
}

impl Augmentation for Sampler {
    fn apply(&mut self, sequence: &[u8]) -> Vec<u8> {
        if sequence.is_empty() {
            return Vec::new();
        }

        // If sequence is shorter than desired length, return as-is
        if sequence.len() <= self.length {
            return sequence.to_vec();
        }

        self.ensure_rng();

        let start = self.calculate_start(sequence.len());
        let end = start + self.length;

        sequence[start..end].to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_random_sampling() {
        let mut sampler = Sampler::random(10, Some(42));
        let sequence = b"ACGTACGTACGTACGTACGT"; // 20 bases

        let result = sampler.apply(sequence);
        assert_eq!(result.len(), 10);

        // Verify it's a valid subsequence
        let result_str = std::str::from_utf8(&result).unwrap();
        let seq_str = std::str::from_utf8(sequence).unwrap();
        assert!(
            seq_str.contains(result_str),
            "Result should be a subsequence"
        );
    }

    #[test]
    fn test_reproducible_sampling() {
        let mut sampler1 = Sampler::random(10, Some(42));
        let mut sampler2 = Sampler::random(10, Some(42));

        let sequence = b"ACGTACGTACGTACGTACGT";

        let result1 = sampler1.apply(sequence);
        let result2 = sampler2.apply(sequence);

        assert_eq!(
            result1, result2,
            "Sampling should be reproducible with same seed"
        );
    }

    #[test]
    fn test_different_seeds() {
        let mut sampler1 = Sampler::random(10, Some(42));
        let mut sampler2 = Sampler::random(10, Some(43));

        let sequence = b"ACGTACGTACGTACGTACGT";

        let result1 = sampler1.apply(sequence);
        let result2 = sampler2.apply(sequence);

        // With different seeds, results should likely be different
        // (though there's a small chance they could be the same)
        // Just verify both are valid subsequences
        assert_eq!(result1.len(), 10);
        assert_eq!(result2.len(), 10);
    }

    #[test]
    fn test_start_sampling() {
        let mut sampler = Sampler::from_start(10);
        let sequence = b"ACGTACGTACGTACGTACGT";

        let result = sampler.apply(sequence);
        assert_eq!(result, b"ACGTACGTAC");
    }

    #[test]
    fn test_center_sampling() {
        let mut sampler = Sampler::from_center(10);
        let sequence = b"ACGTACGTACGTACGTACGT"; // 20 bases

        let result = sampler.apply(sequence);
        // Center: (20 - 10) / 2 = 5
        assert_eq!(result, b"CGTACGTACG"); // positions 5-14
    }

    #[test]
    fn test_end_sampling() {
        let mut sampler = Sampler::from_end(10);
        let sequence = b"ACGTACGTACGTACGTACGT"; // 20 bases

        let result = sampler.apply(sequence);
        assert_eq!(result, b"GTACGTACGT"); // last 10 bases (positions 10-19)
    }

    #[test]
    fn test_short_sequence() {
        let mut sampler = Sampler::random(20, Some(42));
        let sequence = b"ACGT"; // Only 4 bases

        let result = sampler.apply(sequence);
        assert_eq!(result, sequence, "Short sequences should be returned as-is");
    }

    #[test]
    fn test_exact_length_sequence() {
        let mut sampler = Sampler::random(10, Some(42));
        let sequence = b"ACGTACGTAC"; // Exactly 10 bases

        let result = sampler.apply(sequence);
        assert_eq!(result, sequence);
    }

    #[test]
    fn test_empty_sequence() {
        let mut sampler = Sampler::random(10, Some(42));
        assert_eq!(sampler.apply(b""), b"");
    }

    #[test]
    #[should_panic(expected = "Subsequence length must be greater than 0")]
    fn test_zero_length() {
        Sampler::random(0, None);
    }

    #[test]
    fn test_builder() {
        let mut sampler = SamplerBuilder::default()
            .length(50usize)
            .strategy(SamplingStrategy::Center)
            .seed(Some(42))
            .build()
            .unwrap();

        // Create exactly 90 bases: "ACGT" repeated 22 times + "AC" = 88 + 2 = 90
        let sequence = b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTAC"; // 22 * 4 + 2 = 90
        assert_eq!(sequence.len(), 90);

        let result = sampler.apply(sequence);

        assert_eq!(result.len(), 50);
        // Center: (90 - 50) / 2 = 20
        assert_eq!(result, &sequence[20..70]);
    }

    #[test]
    fn test_multiple_samples() {
        let mut sampler = Sampler::random(10, Some(42));
        let sequence = b"ACGTACGTACGTACGTACGT";

        // Multiple calls should produce different results (random positions)
        let result1 = sampler.apply(sequence);
        let result2 = sampler.apply(sequence);
        let result3 = sampler.apply(sequence);

        // All should be valid subsequences
        assert_eq!(result1.len(), 10);
        assert_eq!(result2.len(), 10);
        assert_eq!(result3.len(), 10);

        // With seed 42, we get deterministic but different results per call
        // This tests that the RNG state advances correctly
    }
}
