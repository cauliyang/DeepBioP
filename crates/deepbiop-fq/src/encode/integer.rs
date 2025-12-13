//! Integer encoding implementation for FASTQ sequences.
//!
//! This module provides simple integer encoding where each nucleotide or amino acid
//! is mapped to a unique integer value (e.g., A=0, C=1, G=2, T=3 for DNA).

use anyhow::Result;
use ndarray::{Array1, Array2};
use rayon::prelude::*;

use deepbiop_core::error::DPError;
use deepbiop_core::types::EncodingType;

/// Integer encoder for biological sequences.
///
/// Encodes sequences as integer arrays where each position contains an integer
/// representing the nucleotide/amino acid at that position.
/// For DNA: A=0, C=1, G=2, T=3
/// For RNA: A=0, C=1, G=2, U=3
/// For Protein: A=0, C=1, D=2, E=3, ... (alphabetically)
///
/// # Examples
///
/// ```no_run
/// use deepbiop_fq::encode::integer::IntegerEncoder;
/// use deepbiop_core::types::EncodingType;
/// use ndarray::arr1;
///
/// let encoder = IntegerEncoder::new(EncodingType::DNA);
/// let encoded = encoder.encode(b"ACGT").unwrap();
/// assert_eq!(encoded, arr1(&[0.0, 1.0, 2.0, 3.0]));
/// ```
pub struct IntegerEncoder {
    /// Type of sequence being encoded (DNA, RNA, or Protein)
    encoding_type: EncodingType,
}

impl IntegerEncoder {
    /// Create a new integer encoder.
    ///
    /// # Arguments
    ///
    /// * `encoding_type` - The type of sequence (DNA, RNA, or Protein)
    ///
    /// # Returns
    ///
    /// A new `IntegerEncoder` instance
    pub fn new(encoding_type: EncodingType) -> Self {
        Self { encoding_type }
    }

    /// Encode a single sequence as an integer array.
    ///
    /// # Arguments
    ///
    /// * `sequence` - The sequence to encode
    ///
    /// # Returns
    ///
    /// A 1D array of integers representing the sequence
    ///
    /// # Errors
    ///
    /// Returns an error if the sequence contains invalid characters
    pub fn encode(&self, sequence: &[u8]) -> Result<Array1<f32>> {
        let alphabet = self.encoding_type.alphabet();
        let mut encoded = Array1::<f32>::zeros(sequence.len());

        for (i, &base) in sequence.iter().enumerate() {
            let base_upper = base.to_ascii_uppercase();

            // Find the position of this base in the alphabet
            if let Some(pos) = alphabet.iter().position(|&a| a == base_upper) {
                encoded[i] = pos as f32;
            } else {
                return Err(DPError::InvalidAlphabet {
                    character: base as char,
                    position: i,
                    expected: String::from_utf8_lossy(alphabet).to_string(),
                }
                .into());
            }
        }

        Ok(encoded)
    }

    /// Encode multiple sequences in parallel as integer arrays.
    ///
    /// All sequences are padded to the length of the longest sequence with -1.
    ///
    /// # Arguments
    ///
    /// * `sequences` - Slice of sequences to encode
    ///
    /// # Returns
    ///
    /// A 2D array where each row is an integer-encoded sequence
    ///
    /// # Errors
    ///
    /// Returns an error if any sequence fails to encode
    pub fn encode_batch(&self, sequences: &[&[u8]]) -> Result<Array2<f32>> {
        if sequences.is_empty() {
            return Ok(Array2::zeros((0, 0)));
        }

        // Find the maximum sequence length
        let max_len = sequences.iter().map(|s| s.len()).max().unwrap_or(0);

        // Encode all sequences in parallel
        let encoded_seqs: Result<Vec<Array1<f32>>> =
            sequences.par_iter().map(|seq| self.encode(seq)).collect();

        let encoded_seqs = encoded_seqs?;

        // Create the output array with padding value -1
        let mut batch = Array2::<f32>::from_elem((sequences.len(), max_len), -1.0);

        // Copy each encoded sequence into the batch array
        for (batch_idx, encoded) in encoded_seqs.iter().enumerate() {
            let seq_len = encoded.len();
            for i in 0..seq_len {
                batch[[batch_idx, i]] = encoded[i];
            }
        }

        Ok(batch)
    }

    /// Get the encoding type.
    pub fn encoding_type(&self) -> EncodingType {
        self.encoding_type
    }
}

// Implement SequenceEncoder trait for IntegerEncoder
impl deepbiop_core::encoder::SequenceEncoder for IntegerEncoder {
    type EncodeOutput = Array1<f32>;

    fn encode_sequence(&self, seq: &[u8], qual: Option<&[u8]>) -> Result<Self::EncodeOutput> {
        // Validate inputs
        self.validate_input(seq, qual)?;

        // IntegerEncoder ignores quality scores, only encodes sequence
        self.encode(seq)
    }

    fn expected_output_size(&self, seq_len: usize) -> usize {
        seq_len
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_integer_encode_dna() {
        let encoder = IntegerEncoder::new(EncodingType::DNA);
        let encoded = encoder.encode(b"ACGT").unwrap();

        assert_eq!(encoded.len(), 4);
        assert_eq!(encoded[0], 0.0); // A
        assert_eq!(encoded[1], 1.0); // C
        assert_eq!(encoded[2], 2.0); // G
        assert_eq!(encoded[3], 3.0); // T
    }

    #[test]
    fn test_integer_encode_lowercase() {
        let encoder = IntegerEncoder::new(EncodingType::DNA);
        let encoded = encoder.encode(b"acgt").unwrap();

        assert_eq!(encoded[0], 0.0); // a -> A
        assert_eq!(encoded[1], 1.0); // c -> C
        assert_eq!(encoded[2], 2.0); // g -> G
        assert_eq!(encoded[3], 3.0); // t -> T
    }

    #[test]
    fn test_integer_encode_invalid_char() {
        let encoder = IntegerEncoder::new(EncodingType::DNA);
        let result = encoder.encode(b"ACGTN");
        assert!(result.is_err());
    }

    #[test]
    fn test_integer_encode_rna() {
        let encoder = IntegerEncoder::new(EncodingType::RNA);
        let encoded = encoder.encode(b"ACGU").unwrap();

        assert_eq!(encoded.len(), 4);
        assert_eq!(encoded[0], 0.0); // A
        assert_eq!(encoded[1], 1.0); // C
        assert_eq!(encoded[2], 2.0); // G
        assert_eq!(encoded[3], 3.0); // U
    }

    #[test]
    fn test_integer_encode_protein() {
        let encoder = IntegerEncoder::new(EncodingType::Protein);
        let encoded = encoder.encode(b"ACDE").unwrap();

        assert_eq!(encoded.len(), 4);
        // Should map to positions in alphabetically sorted amino acids
        assert!(encoded[0] >= 0.0);
        assert!(encoded[1] >= 0.0);
        assert!(encoded[2] >= 0.0);
        assert!(encoded[3] >= 0.0);
    }

    #[test]
    fn test_integer_encode_batch() {
        let encoder = IntegerEncoder::new(EncodingType::DNA);
        let sequences = vec![b"ACGT".as_ref(), b"AC".as_ref(), b"ACGTACGT".as_ref()];

        let batch = encoder.encode_batch(&sequences).unwrap();

        assert_eq!(batch.shape(), &[3, 8]); // 3 sequences, max length 8

        // First sequence
        assert_eq!(batch[[0, 0]], 0.0); // A
        assert_eq!(batch[[0, 1]], 1.0); // C

        // Second sequence (shorter, should be padded)
        assert_eq!(batch[[1, 0]], 0.0); // A
        assert_eq!(batch[[1, 1]], 1.0); // C
        assert_eq!(batch[[1, 2]], -1.0); // Padding
    }

    #[test]
    fn test_integer_encode_empty_sequence() {
        let encoder = IntegerEncoder::new(EncodingType::DNA);
        let encoded = encoder.encode(b"").unwrap();
        assert_eq!(encoded.len(), 0);
    }

    #[test]
    fn test_integer_encode_empty_batch() {
        let encoder = IntegerEncoder::new(EncodingType::DNA);
        let sequences: Vec<&[u8]> = vec![];

        let batch = encoder.encode_batch(&sequences).unwrap();
        assert_eq!(batch.shape(), &[0, 0]);
    }
}

// Python bindings
#[cfg(feature = "python")]
pub mod python {
    use super::*;
    use numpy::{PyArray1, PyArray2};
    use pyo3::prelude::*;
    use pyo3_stub_gen::derive::*;

    /// Python wrapper for IntegerEncoder.
    ///
    /// Encodes biological sequences as integer arrays (A=0, C=1, G=2, T/U=3).
    #[gen_stub_pyclass]
    #[pyclass(name = "IntegerEncoder", module = "deepbiop.fq")]
    pub struct PyIntegerEncoder {
        inner: IntegerEncoder,
    }

    #[gen_stub_pymethods]
    #[pymethods]
    impl PyIntegerEncoder {
        /// Create a new integer encoder.
        ///
        /// Args:
        ///     encoding_type: Type of sequence ("dna", "rna", or "protein")
        ///
        /// Returns:
        ///     A new IntegerEncoder instance
        #[new]
        pub fn new(encoding_type: &str) -> PyResult<Self> {
            let enc_type = encoding_type
                .parse::<EncodingType>()
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

            Ok(Self {
                inner: IntegerEncoder::new(enc_type),
            })
        }

        /// Encode a single sequence as an integer array.
        ///
        /// Args:
        ///     sequence: The sequence to encode (bytes)
        ///
        /// Returns:
        ///     NumPy array of shape [sequence_length] with integer values
        ///
        /// Raises:
        ///     ValueError: If the sequence contains invalid characters
        #[pyo3(name = "encode")]
        pub fn encode<'py>(
            &self,
            py: Python<'py>,
            sequence: Vec<u8>,
        ) -> PyResult<Bound<'py, PyArray1<f32>>> {
            let encoded = self
                .inner
                .encode(&sequence)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

            Ok(PyArray1::from_array(py, &encoded))
        }

        /// Encode multiple sequences in parallel.
        ///
        /// All sequences are padded to the length of the longest sequence with -1.
        ///
        /// Args:
        ///     sequences: List of sequences to encode (list of bytes)
        ///
        /// Returns:
        ///     NumPy array of shape [num_sequences, max_length]
        ///
        /// Raises:
        ///     ValueError: If any sequence fails to encode
        #[pyo3(name = "encode_batch")]
        pub fn encode_batch<'py>(
            &self,
            py: Python<'py>,
            sequences: Vec<Vec<u8>>,
        ) -> PyResult<Bound<'py, PyArray2<f32>>> {
            let seq_refs: Vec<&[u8]> = sequences.iter().map(|s| s.as_slice()).collect();

            // Release GIL for parallel processing with Rayon
            let encoded = py
                .detach(|| self.inner.encode_batch(&seq_refs))
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

            Ok(PyArray2::from_array(py, &encoded))
        }

        /// Get the encoding type.
        #[pyo3(name = "encoding_type")]
        pub fn encoding_type(&self) -> String {
            match self.inner.encoding_type() {
                EncodingType::DNA => "dna".to_string(),
                EncodingType::RNA => "rna".to_string(),
                EncodingType::Protein => "protein".to_string(),
            }
        }

        /// String representation.
        #[pyo3(name = "__repr__")]
        pub fn repr(&self) -> String {
            format!("IntegerEncoder(encoding_type='{}')", self.encoding_type())
        }
    }
}
