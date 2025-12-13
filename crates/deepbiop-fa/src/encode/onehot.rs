//! One-hot encoding implementation for FASTA sequences.
//!
//! This module provides one-hot encoding for FASTA sequences, supporting DNA, RNA, and protein sequences.

use anyhow::Result;
use ndarray::{Array2, Array3};
use rayon::prelude::*;

use deepbiop_core::error::DPError;
use deepbiop_core::types::EncodingType;

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
/// use deepbiop_fa::encode::onehot::{OneHotEncoder, AmbiguousStrategy};
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
    #[allow(dead_code)]
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
                        // TODO: Implement random replacement
                        // For now, mask with zeros
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

// Implement SequenceEncoder trait for OneHotEncoder
impl deepbiop_core::encoder::SequenceEncoder for OneHotEncoder {
    type EncodeOutput = Array2<f32>;

    fn encode_sequence(&self, seq: &[u8], qual: Option<&[u8]>) -> Result<Self::EncodeOutput> {
        // Validate inputs
        self.validate_input(seq, qual)?;

        // OneHotEncoder ignores quality scores for FASTA (qual should be None)
        self.encode(seq)
    }

    fn expected_output_size(&self, seq_len: usize) -> usize {
        seq_len * self.encoding_type.alphabet_size()
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

        // C = [0,1,0,0]
        assert_eq!(encoded[[1, 1]], 1.0);

        // G = [0,0,1,0]
        assert_eq!(encoded[[2, 2]], 1.0);

        // T = [0,0,0,1]
        assert_eq!(encoded[[3, 3]], 1.0);
    }

    #[test]
    fn test_onehot_encode_protein() {
        let encoder = OneHotEncoder::new(EncodingType::Protein, AmbiguousStrategy::Skip);
        let encoded = encoder.encode(b"ACDE").unwrap();

        assert_eq!(encoded.shape(), &[4, 20]); // 4 amino acids, 20 possible
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

        // First sequence (ACGT) - first 4 positions should be encoded
        assert_eq!(batch[[0, 0, 0]], 1.0); // A
        assert_eq!(batch[[0, 1, 1]], 1.0); // C

        // Second sequence (AC) - first 2 positions encoded, rest zeros
        assert_eq!(batch[[1, 0, 0]], 1.0); // A
        assert_eq!(batch[[1, 1, 1]], 1.0); // C
        assert_eq!(batch[[1, 2, 0]], 0.0); // Padding
    }
}
