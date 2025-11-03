//! One-hot encoding implementation for FASTQ sequences.

use anyhow::Result;
use ndarray::{Array2, Array3};
use rand::prelude::*;
use rand::rngs::StdRng;
use rand::{rng, SeedableRng};
use rayon::prelude::*;

use deepbiop_core::error::DPError;
use deepbiop_core::types::EncodingType;

#[cfg(feature = "python")]
use numpy::{PyArray2, PyArray3};
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3_stub_gen::derive::*;

/// Strategy for handling ambiguous nucleotides in sequences.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AmbiguousStrategy {
    /// Skip sequences containing ambiguous bases
    Skip,
    /// Mask ambiguous bases with zeros
    Mask,
    /// Replace ambiguous bases with random valid bases
    Random,
}

/// One-hot encoder for biological sequences.
///
/// Encodes sequences as binary matrices where each position is represented
/// as a one-hot vector. For DNA: A=[1,0,0,0], C=[0,1,0,0], G=[0,0,1,0], T=[0,0,0,1]
///
/// # Examples
///
/// ```no_run
/// use deepbiop_fq::encode::onehot::{OneHotEncoder, AmbiguousStrategy};
/// use deepbiop_core::types::EncodingType;
///
/// let encoder = OneHotEncoder::new(EncodingType::DNA, AmbiguousStrategy::Skip);
/// let encoded = encoder.encode(b"ACGT").unwrap();
/// assert_eq!(encoded.shape(), &[4, 4]); // 4 bases, 4 possible values
/// ```
pub struct OneHotEncoder {
    /// Type of sequence being encoded (DNA, RNA, or Protein)
    encoding_type: EncodingType,
    /// Strategy for handling ambiguous bases
    ambiguous_strategy: AmbiguousStrategy,
    /// Random number generator seed (for Random strategy)
    seed: Option<u64>,
}

impl OneHotEncoder {
    /// Create a new one-hot encoder.
    ///
    /// # Arguments
    ///
    /// * `encoding_type` - The type of sequence (DNA, RNA, or Protein)
    /// * `ambiguous_strategy` - How to handle ambiguous bases
    ///
    /// # Returns
    ///
    /// A new `OneHotEncoder` instance
    pub fn new(encoding_type: EncodingType, ambiguous_strategy: AmbiguousStrategy) -> Self {
        Self {
            encoding_type,
            ambiguous_strategy,
            seed: None,
        }
    }

    /// Create a new one-hot encoder with a random seed.
    ///
    /// # Arguments
    ///
    /// * `encoding_type` - The type of sequence (DNA, RNA, or Protein)
    /// * `ambiguous_strategy` - How to handle ambiguous bases
    /// * `seed` - Random seed for reproducible random replacements
    pub fn with_seed(
        encoding_type: EncodingType,
        ambiguous_strategy: AmbiguousStrategy,
        seed: u64,
    ) -> Self {
        Self {
            encoding_type,
            ambiguous_strategy,
            seed: Some(seed),
        }
    }

    /// Encode a single sequence as a one-hot matrix.
    ///
    /// # Arguments
    ///
    /// * `sequence` - The sequence to encode
    ///
    /// # Returns
    ///
    /// A 2D array of shape `[sequence_length, alphabet_size]`
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The sequence contains invalid characters for the encoding type
    /// - The sequence contains ambiguous bases and strategy is `Skip`
    pub fn encode(&self, sequence: &[u8]) -> Result<Array2<f32>> {
        let alphabet_size = self.encoding_type.alphabet_size();
        let seq_len = sequence.len();

        let mut encoded = Array2::<f32>::zeros((seq_len, alphabet_size));

        // Initialize RNG for random strategy (only created if needed)
        let mut rng_state: Option<StdRng> =
            if matches!(self.ambiguous_strategy, AmbiguousStrategy::Random) {
                Some(if let Some(seed) = self.seed {
                    StdRng::seed_from_u64(seed)
                } else {
                    // Use rng() to seed a new StdRng
                    StdRng::from_rng(&mut rng())
                })
            } else {
                None
            };

        for (i, &base) in sequence.iter().enumerate() {
            let base_upper = base.to_ascii_uppercase();

            // Check if base is valid
            if !self.encoding_type.is_valid_char(base_upper) {
                match self.ambiguous_strategy {
                    AmbiguousStrategy::Skip => {
                        return Err(DPError::InvalidAlphabet {
                            character: base as char,
                            position: i,
                            expected: String::from_utf8_lossy(self.encoding_type.alphabet())
                                .to_string(),
                        }
                        .into());
                    }
                    AmbiguousStrategy::Mask => {
                        // Leave as zeros
                        continue;
                    }
                    AmbiguousStrategy::Random => {
                        // Replace with a random valid base from the alphabet
                        let alphabet = self.encoding_type.alphabet();
                        if let Some(ref mut rng_ref) = rng_state {
                            let random_idx = rng_ref.random_range(0..alphabet.len());
                            encoded[[i, random_idx]] = 1.0;
                        }
                        continue;
                    }
                }
            }

            // Find the position of this base in the alphabet
            if let Some(pos) = self
                .encoding_type
                .alphabet()
                .iter()
                .position(|&a| a == base_upper)
            {
                encoded[[i, pos]] = 1.0;
            }
        }

        Ok(encoded)
    }

    /// Encode multiple sequences in parallel.
    ///
    /// All sequences are padded to the length of the longest sequence with zeros.
    ///
    /// # Arguments
    ///
    /// * `sequences` - Slice of sequences to encode
    ///
    /// # Returns
    ///
    /// A 3D array of shape `[num_sequences, max_length, alphabet_size]`
    ///
    /// # Errors
    ///
    /// Returns an error if any sequence fails to encode
    pub fn encode_batch(&self, sequences: &[&[u8]]) -> Result<Array3<f32>> {
        if sequences.is_empty() {
            return Ok(Array3::zeros((0, 0, self.encoding_type.alphabet_size())));
        }

        // Find the maximum sequence length
        let max_len = sequences.iter().map(|s| s.len()).max().unwrap_or(0);
        let alphabet_size = self.encoding_type.alphabet_size();

        // Encode all sequences in parallel
        let encoded_seqs: Result<Vec<Array2<f32>>> =
            sequences.par_iter().map(|seq| self.encode(seq)).collect();

        let encoded_seqs = encoded_seqs?;

        // Create the output array
        let mut batch = Array3::<f32>::zeros((sequences.len(), max_len, alphabet_size));

        // Copy each encoded sequence into the batch array
        for (batch_idx, encoded) in encoded_seqs.iter().enumerate() {
            let seq_len = encoded.shape()[0];
            for i in 0..seq_len {
                for j in 0..alphabet_size {
                    batch[[batch_idx, i, j]] = encoded[[i, j]];
                }
            }
        }

        Ok(batch)
    }

    /// Get the encoding type.
    pub fn encoding_type(&self) -> EncodingType {
        self.encoding_type
    }

    /// Get the ambiguous strategy.
    pub fn ambiguous_strategy(&self) -> AmbiguousStrategy {
        self.ambiguous_strategy
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_onehot_encode_dna() {
        let encoder = OneHotEncoder::new(EncodingType::DNA, AmbiguousStrategy::Skip);
        let encoded = encoder.encode(b"ACGT").unwrap();

        assert_eq!(encoded.shape(), &[4, 4]);

        // A = [1,0,0,0]
        assert_eq!(encoded[[0, 0]], 1.0);
        assert_eq!(encoded[[0, 1]], 0.0);
        assert_eq!(encoded[[0, 2]], 0.0);
        assert_eq!(encoded[[0, 3]], 0.0);

        // C = [0,1,0,0]
        assert_eq!(encoded[[1, 0]], 0.0);
        assert_eq!(encoded[[1, 1]], 1.0);
        assert_eq!(encoded[[1, 2]], 0.0);
        assert_eq!(encoded[[1, 3]], 0.0);

        // G = [0,0,1,0]
        assert_eq!(encoded[[2, 0]], 0.0);
        assert_eq!(encoded[[2, 1]], 0.0);
        assert_eq!(encoded[[2, 2]], 1.0);
        assert_eq!(encoded[[2, 3]], 0.0);

        // T = [0,0,0,1]
        assert_eq!(encoded[[3, 0]], 0.0);
        assert_eq!(encoded[[3, 1]], 0.0);
        assert_eq!(encoded[[3, 2]], 0.0);
        assert_eq!(encoded[[3, 3]], 1.0);
    }

    #[test]
    fn test_onehot_encode_lowercase() {
        let encoder = OneHotEncoder::new(EncodingType::DNA, AmbiguousStrategy::Skip);
        let encoded = encoder.encode(b"acgt").unwrap();
        assert_eq!(encoded.shape(), &[4, 4]);
        assert_eq!(encoded[[0, 0]], 1.0); // a -> A
    }

    #[test]
    fn test_onehot_encode_ambiguous_skip() {
        let encoder = OneHotEncoder::new(EncodingType::DNA, AmbiguousStrategy::Skip);
        let result = encoder.encode(b"ACGTN");
        assert!(result.is_err());
    }

    #[test]
    fn test_onehot_encode_ambiguous_mask() {
        let encoder = OneHotEncoder::new(EncodingType::DNA, AmbiguousStrategy::Mask);
        let encoded = encoder.encode(b"ACGTN").unwrap();

        assert_eq!(encoded.shape(), &[5, 4]);
        // N should be all zeros
        assert_eq!(encoded[[4, 0]], 0.0);
        assert_eq!(encoded[[4, 1]], 0.0);
        assert_eq!(encoded[[4, 2]], 0.0);
        assert_eq!(encoded[[4, 3]], 0.0);
    }

    #[test]
    fn test_onehot_encode_batch() {
        let encoder = OneHotEncoder::new(EncodingType::DNA, AmbiguousStrategy::Skip);
        let sequences = vec![b"ACGT".as_ref(), b"AC".as_ref(), b"ACGTACGT".as_ref()];

        let batch = encoder.encode_batch(&sequences).unwrap();

        assert_eq!(batch.shape(), &[3, 8, 4]); // 3 sequences, max length 8, 4 bases

        // First sequence (ACGT) - first 4 positions should be encoded, rest zeros
        assert_eq!(batch[[0, 0, 0]], 1.0); // A
        assert_eq!(batch[[0, 1, 1]], 1.0); // C

        // Second sequence (AC) - first 2 positions encoded, rest zeros
        assert_eq!(batch[[1, 0, 0]], 1.0); // A
        assert_eq!(batch[[1, 1, 1]], 1.0); // C
        assert_eq!(batch[[1, 2, 0]], 0.0); // Padding
    }

    #[test]
    fn test_onehot_encode_rna() {
        let encoder = OneHotEncoder::new(EncodingType::RNA, AmbiguousStrategy::Skip);
        let encoded = encoder.encode(b"ACGU").unwrap();

        assert_eq!(encoded.shape(), &[4, 4]);
        assert_eq!(encoded[[3, 3]], 1.0); // U
    }

    #[test]
    fn test_onehot_encode_protein() {
        let encoder = OneHotEncoder::new(EncodingType::Protein, AmbiguousStrategy::Skip);
        let encoded = encoder.encode(b"ACDE").unwrap();

        assert_eq!(encoded.shape(), &[4, 20]); // 4 amino acids, 20 possible
    }

    #[test]
    fn test_onehot_encode_ambiguous_random() {
        // Test with a seed for reproducibility
        let encoder = OneHotEncoder::with_seed(EncodingType::DNA, AmbiguousStrategy::Random, 42);
        let encoded = encoder.encode(b"ACGTN").unwrap();

        assert_eq!(encoded.shape(), &[5, 4]);

        // First 4 bases should be encoded correctly
        assert_eq!(encoded[[0, 0]], 1.0); // A
        assert_eq!(encoded[[1, 1]], 1.0); // C
        assert_eq!(encoded[[2, 2]], 1.0); // G
        assert_eq!(encoded[[3, 3]], 1.0); // T

        // N should be replaced with a random valid base (exactly one position should be 1.0)
        let n_encoding: Vec<f32> = (0..4).map(|i| encoded[[4, i]]).collect();
        let sum: f32 = n_encoding.iter().sum();
        assert_eq!(sum, 1.0); // Exactly one hot
        assert!(n_encoding.contains(&1.0)); // At least one 1.0
    }

    #[test]
    fn test_onehot_encode_ambiguous_random_reproducible() {
        // Same seed should produce same results
        let encoder1 = OneHotEncoder::with_seed(EncodingType::DNA, AmbiguousStrategy::Random, 42);
        let encoder2 = OneHotEncoder::with_seed(EncodingType::DNA, AmbiguousStrategy::Random, 42);

        let encoded1 = encoder1.encode(b"ACGTN").unwrap();
        let encoded2 = encoder2.encode(b"ACGTN").unwrap();

        // Should produce identical encodings
        assert_eq!(encoded1, encoded2);
    }

    #[test]
    fn test_onehot_encode_ambiguous_random_different_seeds() {
        // Different seeds should likely produce different results
        let encoder1 = OneHotEncoder::with_seed(EncodingType::DNA, AmbiguousStrategy::Random, 42);
        let encoder2 = OneHotEncoder::with_seed(EncodingType::DNA, AmbiguousStrategy::Random, 123);

        let encoded1 = encoder1.encode(b"NNNNN").unwrap();
        let encoded2 = encoder2.encode(b"NNNNN").unwrap();

        // With 5 N's, it's extremely unlikely they'd be encoded identically with different seeds
        // But we'll just check that both are valid encodings (each position has exactly one hot)
        for i in 0..5 {
            let sum1: f32 = (0..4).map(|j| encoded1[[i, j]]).sum();
            let sum2: f32 = (0..4).map(|j| encoded2[[i, j]]).sum();
            assert_eq!(sum1, 1.0);
            assert_eq!(sum2, 1.0);
        }
    }
}

// Python bindings
#[cfg(feature = "python")]
pub mod python {
    use super::*;

    /// Python wrapper for OneHotEncoder.
    ///
    /// Encodes biological sequences as one-hot matrices for machine learning.
    #[gen_stub_pyclass(module = "deepbiop.fq")]
    #[pyclass(name = "OneHotEncoder")]
    pub struct PyOneHotEncoder {
        inner: OneHotEncoder,
    }

    #[gen_stub_pymethods]
    #[pymethods]
    impl PyOneHotEncoder {
        /// Create a new one-hot encoder.
        ///
        /// Args:
        ///     encoding_type: Type of sequence ("dna", "rna", or "protein")
        ///     ambiguous_strategy: How to handle ambiguous bases ("skip", "mask", or "random")
        ///     seed: Optional random seed for reproducible random replacements
        ///
        /// Returns:
        ///     A new OneHotEncoder instance
        #[new]
        #[pyo3(signature = (encoding_type, ambiguous_strategy, seed=None))]
        pub fn new(
            encoding_type: &str,
            ambiguous_strategy: &str,
            seed: Option<u64>,
        ) -> PyResult<Self> {
            let enc_type = encoding_type
                .parse::<EncodingType>()
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

            let amb_strategy = match ambiguous_strategy.to_lowercase().as_str() {
                "skip" => AmbiguousStrategy::Skip,
                "mask" => AmbiguousStrategy::Mask,
                "random" => AmbiguousStrategy::Random,
                _ => {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "Invalid ambiguous strategy: '{}'. Expected 'skip', 'mask', or 'random'",
                        ambiguous_strategy
                    )))
                }
            };

            let inner = if let Some(s) = seed {
                OneHotEncoder::with_seed(enc_type, amb_strategy, s)
            } else {
                OneHotEncoder::new(enc_type, amb_strategy)
            };

            Ok(Self { inner })
        }

        /// Encode a single sequence as a one-hot matrix.
        ///
        /// Args:
        ///     sequence: The sequence to encode (bytes)
        ///
        /// Returns:
        ///     NumPy array of shape [sequence_length, alphabet_size]
        ///
        /// Raises:
        ///     ValueError: If the sequence contains invalid characters
        #[pyo3(name = "encode")]
        pub fn encode<'py>(
            &self,
            py: Python<'py>,
            sequence: Vec<u8>,
        ) -> PyResult<Bound<'py, PyArray2<f32>>> {
            let encoded = self
                .inner
                .encode(&sequence)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

            Ok(PyArray2::from_array(py, &encoded))
        }

        /// Encode multiple sequences in parallel.
        ///
        /// All sequences are padded to the length of the longest sequence with zeros.
        ///
        /// Args:
        ///     sequences: List of sequences to encode (list of bytes)
        ///
        /// Returns:
        ///     NumPy array of shape [num_sequences, max_length, alphabet_size]
        ///
        /// Raises:
        ///     ValueError: If any sequence fails to encode
        #[pyo3(name = "encode_batch")]
        pub fn encode_batch<'py>(
            &self,
            py: Python<'py>,
            sequences: Vec<Vec<u8>>,
        ) -> PyResult<Bound<'py, PyArray3<f32>>> {
            let seq_refs: Vec<&[u8]> = sequences.iter().map(|s| s.as_slice()).collect();
            let encoded = self
                .inner
                .encode_batch(&seq_refs)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

            Ok(PyArray3::from_array(py, &encoded))
        }

        /// Get the encoding type.
        ///
        /// Returns:
        ///     The encoding type as a string ("dna", "rna", or "protein")
        #[pyo3(name = "encoding_type")]
        pub fn encoding_type(&self) -> String {
            match self.inner.encoding_type() {
                EncodingType::DNA => "dna".to_string(),
                EncodingType::RNA => "rna".to_string(),
                EncodingType::Protein => "protein".to_string(),
            }
        }

        /// Get the ambiguous strategy.
        ///
        /// Returns:
        ///     The ambiguous strategy as a string ("skip", "mask", or "random")
        #[pyo3(name = "ambiguous_strategy")]
        pub fn ambiguous_strategy(&self) -> String {
            match self.inner.ambiguous_strategy() {
                AmbiguousStrategy::Skip => "skip".to_string(),
                AmbiguousStrategy::Mask => "mask".to_string(),
                AmbiguousStrategy::Random => "random".to_string(),
            }
        }

        /// String representation.
        #[pyo3(name = "__repr__")]
        pub fn repr(&self) -> String {
            format!(
                "OneHotEncoder(encoding_type='{}', ambiguous_strategy='{}')",
                self.encoding_type(),
                self.ambiguous_strategy()
            )
        }
    }
}
