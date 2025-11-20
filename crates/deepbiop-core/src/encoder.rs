//! Unified sequence encoding interface for FASTA and FASTQ formats
//!
//! This module provides a common trait for encoding biological sequences into
//! numerical representations suitable for machine learning. The trait uses
//! zero-cost abstractions to handle both FASTA (sequence-only) and FASTQ
//! (sequence + quality scores) formats efficiently.

use anyhow::Result;

/// Unified trait for encoding biological sequences
///
/// This trait provides a common interface for sequence encoders that work with
/// both FASTA and FASTQ formats. The key innovation is using `Option<&[u8]>` for
/// quality scores, which allows FASTA encoders to pass `None` and FASTQ encoders
/// to pass `Some(qual)`, all with zero runtime cost through monomorphization.
///
/// # Type Parameters
///
/// * `EncodeOutput` - The output type of the encoding (e.g., `Vec<f32>`, `Array2<u8>`)
///
/// # Design Rationale
///
/// Using `Option<&[u8]>` for quality scores provides several benefits:
/// - Zero-cost abstraction: The compiler optimizes away the Option for FASTA
/// - Unified interface: Same trait works for both FASTA and FASTQ
/// - Type safety: Encoders must explicitly handle the quality score case
/// - Memory efficiency: No allocation overhead for FASTA files
///
/// # Examples
///
/// ## Implementing for a simple encoder
///
/// ```
/// use deepbiop_core::encoder::SequenceEncoder;
/// use anyhow::Result;
///
/// struct SimpleEncoder;
///
/// impl SequenceEncoder for SimpleEncoder {
///     type EncodeOutput = Vec<u8>;
///
///     fn encode_sequence(&self, seq: &[u8], qual: Option<&[u8]>) -> Result<Self::EncodeOutput> {
///         // For FASTA files, qual will be None
///         // For FASTQ files, qual will be Some(quality_scores)
///         Ok(seq.to_vec())
///     }
///
///     fn expected_output_size(&self, seq_len: usize) -> usize {
///         seq_len
///     }
/// }
///
/// let encoder = SimpleEncoder;
///
/// // FASTA-style encoding (no quality scores)
/// let fasta_encoded = encoder.encode_sequence(b"ACGT", None).unwrap();
/// assert_eq!(fasta_encoded, b"ACGT");
///
/// // FASTQ-style encoding (with quality scores)
/// let fastq_encoded = encoder.encode_sequence(b"ACGT", Some(b"IIII")).unwrap();
/// assert_eq!(fastq_encoded, b"ACGT");
/// ```
///
/// ## Implementing an encoder that uses quality scores
///
/// ```
/// use deepbiop_core::encoder::SequenceEncoder;
/// use anyhow::Result;
///
/// struct QualityAwareEncoder;
///
/// impl SequenceEncoder for QualityAwareEncoder {
///     type EncodeOutput = Vec<f32>;
///
///     fn encode_sequence(&self, seq: &[u8], qual: Option<&[u8]>) -> Result<Self::EncodeOutput> {
///         let mut output = Vec::with_capacity(seq.len());
///
///         match qual {
///             Some(quality_scores) => {
///                 // Use quality scores to weight the encoding
///                 for (base, &q) in seq.iter().zip(quality_scores.iter()) {
///                     let weight = (q - 33) as f32 / 40.0; // Phred+33 to [0, 1]
///                     output.push(weight);
///                 }
///             }
///             None => {
///                 // No quality scores, use uniform weights
///                 output.extend(std::iter::repeat(1.0).take(seq.len()));
///             }
///         }
///
///         Ok(output)
///     }
///
///     fn expected_output_size(&self, seq_len: usize) -> usize {
///         seq_len
///     }
/// }
/// ```
pub trait SequenceEncoder {
    /// The type of the encoded output
    ///
    /// Common choices include:
    /// - `Vec<u8>` for integer encoding
    /// - `Vec<f32>` for one-hot or weighted encoding
    /// - `Array2<f32>` for 2D representations
    type EncodeOutput;

    /// Encodes a biological sequence into a numerical representation
    ///
    /// # Arguments
    ///
    /// * `seq` - The nucleotide sequence as a byte slice (e.g., b"ACGT")
    /// * `qual` - Optional quality scores for FASTQ files. Pass `None` for FASTA.
    ///
    /// # Returns
    ///
    /// Returns the encoded representation wrapped in a Result.
    ///
    /// # Errors
    ///
    /// May return an error if:
    /// - The sequence contains invalid nucleotides
    /// - Quality scores length doesn't match sequence length
    /// - Encoding computation fails
    ///
    /// # Implementation Notes
    ///
    /// For FASTA-only encoders, you can ignore the `qual` parameter:
    /// ```ignore
    /// fn encode_sequence(&self, seq: &[u8], _qual: Option<&[u8]>) -> Result<Self::EncodeOutput> {
    ///     // Implement encoding using only seq
    /// }
    /// ```
    ///
    /// For FASTQ-aware encoders, match on `qual`:
    /// ```ignore
    /// fn encode_sequence(&self, seq: &[u8], qual: Option<&[u8]>) -> Result<Self::EncodeOutput> {
    ///     match qual {
    ///         Some(q) => { /* use quality scores */ },
    ///         None => { /* fallback encoding */ },
    ///     }
    /// }
    /// ```
    fn encode_sequence(&self, seq: &[u8], qual: Option<&[u8]>) -> Result<Self::EncodeOutput>;

    /// Returns the expected size of the encoded output
    ///
    /// This helps with pre-allocation and validation. The return value
    /// should represent the number of elements in the output, not bytes.
    ///
    /// # Arguments
    ///
    /// * `seq_len` - The length of the input sequence
    ///
    /// # Examples
    ///
    /// For an integer encoder: `seq_len`
    /// For a one-hot encoder (4 channels): `seq_len * 4`
    /// For a k-mer encoder (k=3): `seq_len - k + 1`
    fn expected_output_size(&self, seq_len: usize) -> usize;

    /// Validates that a sequence and quality scores are compatible
    ///
    /// Default implementation checks that quality scores (if present) have
    /// the same length as the sequence. Override this method if you need
    /// additional validation logic.
    ///
    /// # Arguments
    ///
    /// * `seq` - The nucleotide sequence
    /// * `qual` - Optional quality scores
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if valid, or an error describing the validation failure.
    fn validate_input(&self, seq: &[u8], qual: Option<&[u8]>) -> Result<()> {
        if let Some(q) = qual {
            if seq.len() != q.len() {
                return Err(anyhow::anyhow!(
                    "Sequence length ({}) doesn't match quality scores length ({})",
                    seq.len(),
                    q.len()
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test encoder that simply counts bases
    struct CountEncoder;

    impl SequenceEncoder for CountEncoder {
        type EncodeOutput = usize;

        fn encode_sequence(&self, seq: &[u8], _qual: Option<&[u8]>) -> Result<Self::EncodeOutput> {
            Ok(seq.len())
        }

        fn expected_output_size(&self, seq_len: usize) -> usize {
            seq_len
        }
    }

    #[test]
    fn test_encoder_with_fasta() {
        let encoder = CountEncoder;
        let result = encoder.encode_sequence(b"ACGTACGT", None).unwrap();
        assert_eq!(result, 8);
    }

    #[test]
    fn test_encoder_with_fastq() {
        let encoder = CountEncoder;
        let result = encoder.encode_sequence(b"ACGT", Some(b"IIII")).unwrap();
        assert_eq!(result, 4);
    }

    #[test]
    fn test_validate_input_success() {
        let encoder = CountEncoder;
        assert!(encoder.validate_input(b"ACGT", Some(b"IIII")).is_ok());
        assert!(encoder.validate_input(b"ACGT", None).is_ok());
    }

    #[test]
    fn test_validate_input_length_mismatch() {
        let encoder = CountEncoder;
        let result = encoder.validate_input(b"ACGT", Some(b"III"));
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("doesn't match"));
    }

    #[test]
    fn test_expected_output_size() {
        let encoder = CountEncoder;
        assert_eq!(encoder.expected_output_size(100), 100);
    }

    // Quality-aware encoder for more comprehensive testing
    struct WeightedEncoder;

    impl SequenceEncoder for WeightedEncoder {
        type EncodeOutput = Vec<f32>;

        fn encode_sequence(&self, seq: &[u8], qual: Option<&[u8]>) -> Result<Self::EncodeOutput> {
            self.validate_input(seq, qual)?;

            let output = match qual {
                Some(q) => seq
                    .iter()
                    .zip(q.iter())
                    .map(|(_, &quality)| (quality - 33) as f32 / 40.0)
                    .collect(),
                None => vec![1.0; seq.len()],
            };

            Ok(output)
        }

        fn expected_output_size(&self, seq_len: usize) -> usize {
            seq_len
        }
    }

    #[test]
    fn test_weighted_encoder_with_quality() {
        let encoder = WeightedEncoder;
        let result = encoder.encode_sequence(b"ACGT", Some(b"IIII")).unwrap();
        assert_eq!(result.len(), 4);
        // 'I' has ASCII 73, Phred+33 score of 40
        let expected = 40.0 / 40.0;
        for &weight in &result {
            assert!((weight - expected).abs() < 0.001);
        }
    }

    #[test]
    fn test_weighted_encoder_without_quality() {
        let encoder = WeightedEncoder;
        let result = encoder.encode_sequence(b"ACGT", None).unwrap();
        assert_eq!(result.len(), 4);
        for &weight in &result {
            assert!((weight - 1.0).abs() < 0.001);
        }
    }
}
