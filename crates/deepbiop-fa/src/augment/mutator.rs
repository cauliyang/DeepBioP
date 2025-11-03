//! Random mutation augmentation for sequences.

use super::Augmentation;
use derive_builder::Builder;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Random point mutation augmenter.
///
/// Applies random base substitutions at a configurable rate.
/// Useful for improving model robustness to sequencing errors.
///
/// # Examples
///
/// ```
/// use deepbiop_fq::augment::{Mutator, Augmentation};
///
/// // 1% mutation rate with seed for reproducibility
/// let mut mutator = Mutator::new(0.01, Some(42));
/// let sequence = b"AAAAAAAAAA"; // 10 bases
/// let mutated = mutator.apply(sequence);
/// // Expect ~0-2 mutations with 1% rate
/// ```
#[derive(Debug, Clone, Builder)]
#[builder(setter(into))]
pub struct Mutator {
    /// Mutation rate (0.0 to 1.0)
    #[builder(default = "0.01")]
    mutation_rate: f64,

    /// Random seed for reproducibility
    #[builder(default = "None")]
    seed: Option<u64>,

    /// Possible bases for mutations (default: DNA)
    #[builder(default = "vec![b'A', b'C', b'G', b'T']")]
    alphabet: Vec<u8>,

    /// Internal RNG
    #[builder(setter(skip), default = "None")]
    rng: Option<StdRng>,
}

impl Mutator {
    /// Create a new mutator with specified rate and optional seed.
    ///
    /// # Arguments
    ///
    /// * `mutation_rate` - Probability of mutating each base (0.0 to 1.0)
    /// * `seed` - Optional seed for reproducible mutations
    ///
    /// # Panics
    ///
    /// Panics if mutation_rate is not between 0.0 and 1.0
    pub fn new(mutation_rate: f64, seed: Option<u64>) -> Self {
        assert!(
            (0.0..=1.0).contains(&mutation_rate),
            "Mutation rate must be between 0.0 and 1.0"
        );

        Self {
            mutation_rate,
            seed,
            alphabet: vec![b'A', b'C', b'G', b'T'],
            rng: None,
        }
    }

    /// Create a mutator for RNA sequences.
    pub fn for_rna(mutation_rate: f64, seed: Option<u64>) -> Self {
        assert!(
            (0.0..=1.0).contains(&mutation_rate),
            "Mutation rate must be between 0.0 and 1.0"
        );

        Self {
            mutation_rate,
            seed,
            alphabet: vec![b'A', b'C', b'G', b'U'],
            rng: None,
        }
    }

    /// Set custom alphabet for mutations.
    pub fn with_alphabet(mut self, alphabet: Vec<u8>) -> Self {
        assert!(!alphabet.is_empty(), "Alphabet cannot be empty");
        self.alphabet = alphabet;
        self
    }

    /// Initialize RNG if needed (lazy initialization).
    fn ensure_rng(&mut self) {
        if self.rng.is_none() {
            self.rng = Some(if let Some(seed) = self.seed {
                StdRng::seed_from_u64(seed)
            } else {
                StdRng::from_rng(&mut rand::rng())
            });
        }
    }

    /// Mutate a single base to a different base from the alphabet.
    fn mutate_base(&mut self, original: u8) -> u8 {
        let rng = self.rng.as_mut().expect("RNG not initialized");

        // Get possible mutations (exclude the original base)
        let candidates: Vec<u8> = self
            .alphabet
            .iter()
            .copied()
            .filter(|&b| !b.eq_ignore_ascii_case(&original))
            .collect();

        if candidates.is_empty() {
            // If no candidates (e.g., alphabet is just one base), return original
            return original;
        }

        // Select random mutation
        let idx = rng.random_range(0..candidates.len());
        candidates[idx]
    }
}

impl Augmentation for Mutator {
    fn apply(&mut self, sequence: &[u8]) -> Vec<u8> {
        self.ensure_rng();

        sequence
            .iter()
            .map(|&base| {
                let should_mutate = {
                    let rng = self.rng.as_mut().expect("RNG not initialized");
                    rng.random_range(0.0..1.0) < self.mutation_rate
                };

                if should_mutate {
                    self.mutate_base(base)
                } else {
                    base
                }
            })
            .collect()
    }
}

impl Mutator {
    /// Apply mutations to a batch of sequences in parallel.
    ///
    /// Each sequence in the batch is mutated independently using a derived seed
    /// (if the mutator was created with a seed).
    ///
    /// # Arguments
    ///
    /// * `sequences` - Slice of sequences to mutate
    ///
    /// # Returns
    ///
    /// Vector of mutated sequences
    ///
    /// # Example
    ///
    /// ```
    /// use deepbiop_fq::augment::Mutator;
    ///
    /// let mut mutator = Mutator::new(0.01, Some(42));
    /// let sequences = vec![b"ACGTACGT".to_vec(), b"TTAATTAA".to_vec()];
    /// let results = mutator.apply_batch(&sequences);
    /// assert_eq!(results.len(), 2);
    /// ```
    pub fn apply_batch(&self, sequences: &[Vec<u8>]) -> Vec<Vec<u8>> {
        use rayon::prelude::*;

        sequences
            .par_iter()
            .enumerate()
            .map(|(idx, seq)| {
                // Create a derived seed for reproducibility
                let derived_seed = self.seed.map(|s| s.wrapping_add(idx as u64));

                // Create a new mutator with derived seed for this sequence
                let mut local_mutator = Mutator {
                    mutation_rate: self.mutation_rate,
                    seed: derived_seed,
                    alphabet: self.alphabet.clone(),
                    rng: None,
                };

                local_mutator.ensure_rng();

                // Apply mutation
                seq.iter()
                    .map(|&base| {
                        let should_mutate = {
                            let rng = local_mutator.rng.as_mut().expect("RNG not initialized");
                            rng.random_range(0.0..1.0) < local_mutator.mutation_rate
                        };

                        if should_mutate {
                            local_mutator.mutate_base(base)
                        } else {
                            base
                        }
                    })
                    .collect()
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_mutation_rate() {
        let mut mutator = Mutator::new(0.0, Some(42));
        let sequence = b"ACGTACGT";
        let result = mutator.apply(sequence);
        assert_eq!(result, sequence);
    }

    #[test]
    fn test_reproducible_mutations() {
        let mut mutator1 = Mutator::new(0.1, Some(42));
        let mut mutator2 = Mutator::new(0.1, Some(42));

        let sequence = b"AAAAAAAAAA";

        let result1 = mutator1.apply(sequence);
        let result2 = mutator2.apply(sequence);

        assert_eq!(
            result1, result2,
            "Mutations should be reproducible with same seed"
        );
    }

    #[test]
    fn test_high_mutation_rate() {
        let mut mutator = Mutator::new(0.9, Some(42));
        let sequence = b"AAAAAAAAAA"; // 10 A's

        let result = mutator.apply(sequence);

        // Count how many bases were mutated
        let mutations = sequence
            .iter()
            .zip(result.iter())
            .filter(|(a, b)| a != b)
            .count();

        // With 90% mutation rate, expect 7-10 mutations in 10 bases
        assert!(
            mutations >= 7,
            "Expected at least 7 mutations, got {}",
            mutations
        );
    }

    #[test]
    fn test_mutations_change_bases() {
        let mut mutator = Mutator::new(1.0, Some(42)); // 100% mutation
        let sequence = b"AAAA";

        let result = mutator.apply(sequence);

        // All bases should be mutated to something different
        for (original, mutated) in sequence.iter().zip(result.iter()) {
            assert_ne!(original, mutated, "Base should be mutated");
            assert!(
                [b'C', b'G', b'T'].contains(mutated),
                "Mutated base should be C, G, or T, got {}",
                *mutated as char
            );
        }
    }

    #[test]
    fn test_rna_mode() {
        let mut mutator = Mutator::for_rna(1.0, Some(42));
        let sequence = b"AAAA";

        let result = mutator.apply(sequence);

        // All mutations should be to RNA bases
        for &base in &result {
            assert!(
                [b'A', b'C', b'G', b'U'].contains(&base),
                "Expected RNA base, got {}",
                base as char
            );
        }
    }

    #[test]
    fn test_custom_alphabet() {
        let mut mutator = Mutator::new(1.0, Some(42)).with_alphabet(vec![b'A', b'B']);

        let sequence = b"AAAA";
        let result = mutator.apply(sequence);

        // All mutations should be to B (only other option)
        assert_eq!(result, b"BBBB");
    }

    #[test]
    fn test_empty_sequence() {
        let mut mutator = Mutator::new(0.5, Some(42));
        assert_eq!(mutator.apply(b""), b"");
    }

    #[test]
    #[should_panic(expected = "Mutation rate must be between 0.0 and 1.0")]
    fn test_invalid_mutation_rate() {
        Mutator::new(1.5, None);
    }

    #[test]
    fn test_builder() {
        let mut mutator = MutatorBuilder::default()
            .mutation_rate(0.05)
            .seed(Some(42))
            .alphabet(vec![b'A', b'C', b'G', b'T'])
            .build()
            .unwrap();

        let sequence = b"ACGTACGT";
        let result = mutator.apply(sequence);

        // Just verify it compiles and runs
        assert_eq!(result.len(), sequence.len());
    }

    #[test]
    fn test_batch_processing() {
        let mutator = Mutator::new(0.05, Some(42));
        let sequences = vec![
            b"ACGTACGTACGT".to_vec(),
            b"TTAATTAATTAA".to_vec(),
            b"GGCCGGCCGGCC".to_vec(),
        ];

        let results = mutator.apply_batch(&sequences);

        assert_eq!(results.len(), 3);
        // Each result should have the same length as input
        for (i, result) in results.iter().enumerate() {
            assert_eq!(result.len(), sequences[i].len());
        }
    }

    #[test]
    fn test_batch_reproducibility() {
        let mutator1 = Mutator::new(0.1, Some(12345));
        let mutator2 = Mutator::new(0.1, Some(12345));

        let sequences = vec![b"ACGTACGT".to_vec(), b"TTAATTAA".to_vec()];

        let results1 = mutator1.apply_batch(&sequences);
        let results2 = mutator2.apply_batch(&sequences);

        assert_eq!(
            results1, results2,
            "Batch mutations should be reproducible with same seed"
        );
    }
}
