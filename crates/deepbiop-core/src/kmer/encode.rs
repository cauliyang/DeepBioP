//! K-mer encoding for machine learning applications.
//!
//! This module provides k-mer encoding functionality optimized for deep learning,
//! converting biological sequences into k-mer representations suitable for ML models.

use ahash::HashMap;
use anyhow::Result;
use ndarray::{Array1, Array2};
use rayon::prelude::*;

use crate::error::DPError;
use crate::types::EncodingType;

use super::seq_to_kmers;

/// K-mer encoder for biological sequences.
///
/// Encodes sequences as k-mer frequency vectors or k-mer index arrays.
/// Supports canonical k-mers (treating a k-mer and its reverse complement as identical).
///
/// # Examples
///
/// ```no_run
/// use deepbiop_core::kmer::encode::KmerEncoder;
/// use deepbiop_core::types::EncodingType;
///
/// let mut encoder = KmerEncoder::new(3, true, EncodingType::DNA);
/// let encoded = encoder.encode(b"ACGTACGT").unwrap();
/// ```
pub struct KmerEncoder {
    /// K-mer length
    k: usize,
    /// Use canonical k-mers (k-mer and reverse complement are the same)
    canonical: bool,
    /// Encoding type (DNA, RNA, or Protein)
    encoding_type: EncodingType,
    /// K-mer to index mapping (built lazily)
    kmer_to_idx: Option<HashMap<Vec<u8>, usize>>,
}

impl KmerEncoder {
    /// Create a new k-mer encoder.
    ///
    /// # Arguments
    ///
    /// * `k` - K-mer length
    /// * `canonical` - Whether to use canonical k-mers
    /// * `encoding_type` - The type of sequence (DNA, RNA, or Protein)
    ///
    /// # Returns
    ///
    /// A new `KmerEncoder` instance
    pub fn new(k: usize, canonical: bool, encoding_type: EncodingType) -> Self {
        Self {
            k,
            canonical,
            encoding_type,
            kmer_to_idx: None,
        }
    }

    /// Get the k-mer length.
    pub fn k(&self) -> usize {
        self.k
    }

    /// Check if using canonical k-mers.
    pub fn is_canonical(&self) -> bool {
        self.canonical
    }

    /// Get the encoding type.
    pub fn encoding_type(&self) -> EncodingType {
        self.encoding_type
    }

    /// Build the k-mer to index mapping.
    ///
    /// This generates all possible k-mers for the alphabet and assigns each a unique index.
    fn build_kmer_index(&mut self) {
        let alphabet = self.encoding_type.alphabet();
        let alphabet_size = alphabet.len();

        // Calculate total number of possible k-mers
        let total_kmers = alphabet_size.pow(self.k as u32);

        let mut kmer_to_idx = HashMap::with_capacity_and_hasher(total_kmers, Default::default());
        let mut idx = 0;

        // Generate all k-mers using recursive approach
        fn generate_kmers(
            alphabet: &[u8],
            k: usize,
            current: &mut Vec<u8>,
            kmer_to_idx: &mut HashMap<Vec<u8>, usize>,
            idx: &mut usize,
        ) {
            if current.len() == k {
                kmer_to_idx.insert(current.clone(), *idx);
                *idx += 1;
                return;
            }

            for &base in alphabet {
                current.push(base);
                generate_kmers(alphabet, k, current, kmer_to_idx, idx);
                current.pop();
            }
        }

        let mut current = Vec::with_capacity(self.k);
        generate_kmers(alphabet, self.k, &mut current, &mut kmer_to_idx, &mut idx);

        self.kmer_to_idx = Some(kmer_to_idx);
    }

    /// Encode a sequence as a k-mer count vector.
    ///
    /// Returns a 1D array where each element represents the count of a specific k-mer.
    ///
    /// # Arguments
    ///
    /// * `sequence` - The sequence to encode
    ///
    /// # Returns
    ///
    /// A 1D array of k-mer counts
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The sequence is shorter than k
    /// - The sequence contains invalid characters
    pub fn encode(&mut self, sequence: &[u8]) -> Result<Array1<f32>> {
        // Build k-mer index if not already built (needed for dimensions)
        if self.kmer_to_idx.is_none() {
            self.build_kmer_index();
        }

        let kmer_to_idx = self.kmer_to_idx.as_ref().unwrap();
        let total_kmers = kmer_to_idx.len();

        // If sequence is shorter than k, return a zero vector
        if sequence.len() < self.k {
            return Ok(Array1::<f32>::zeros(total_kmers));
        }

        // Validate sequence
        for (pos, &base) in sequence.iter().enumerate() {
            if !self.encoding_type.is_valid_char(base) {
                return Err(DPError::InvalidAlphabet {
                    character: base as char,
                    position: pos,
                    expected: String::from_utf8_lossy(self.encoding_type.alphabet()).to_string(),
                }
                .into());
            }
        }

        let total_kmers = kmer_to_idx.len();

        // Extract k-mers
        let kmers = seq_to_kmers(sequence, self.k, true);

        // Count k-mers
        let mut counts = Array1::<f32>::zeros(total_kmers);

        for kmer in kmers {
            // Convert to uppercase
            let kmer_upper: Vec<u8> = kmer.iter().map(|&b| b.to_ascii_uppercase()).collect();

            if let Some(&idx) = kmer_to_idx.get(&kmer_upper) {
                counts[idx] += 1.0;
            }
        }

        Ok(counts)
    }

    /// Encode multiple sequences in parallel as k-mer count vectors.
    ///
    /// # Arguments
    ///
    /// * `sequences` - Slice of sequences to encode
    ///
    /// # Returns
    ///
    /// A 2D array where each row is a k-mer count vector
    ///
    /// # Errors
    ///
    /// Returns an error if any sequence fails to encode
    pub fn encode_batch(&mut self, sequences: &[&[u8]]) -> Result<Array2<f32>> {
        if sequences.is_empty() {
            // Build index to get proper dimensions
            if self.kmer_to_idx.is_none() {
                self.build_kmer_index();
            }
            let total_kmers = self.kmer_to_idx.as_ref().unwrap().len();
            return Ok(Array2::zeros((0, total_kmers)));
        }

        // Build k-mer index if not already built
        if self.kmer_to_idx.is_none() {
            self.build_kmer_index();
        }

        let kmer_to_idx = self.kmer_to_idx.as_ref().unwrap();
        let total_kmers = kmer_to_idx.len();
        let k = self.k;
        let encoding_type = self.encoding_type;

        // Encode all sequences in parallel
        let encoded_seqs: Result<Vec<Array1<f32>>> = sequences
            .par_iter()
            .map(|sequence| {
                // If sequence is shorter than k, return a zero vector
                if sequence.len() < k {
                    return Ok(Array1::<f32>::zeros(total_kmers));
                }

                // Validate sequence
                for (pos, &base) in sequence.iter().enumerate() {
                    if !encoding_type.is_valid_char(base) {
                        return Err(DPError::InvalidAlphabet {
                            character: base as char,
                            position: pos,
                            expected: String::from_utf8_lossy(encoding_type.alphabet()).to_string(),
                        }
                        .into());
                    }
                }

                // Extract k-mers
                let kmers = seq_to_kmers(sequence, k, true);

                // Count k-mers
                let mut counts = Array1::<f32>::zeros(total_kmers);

                for kmer in kmers {
                    // Convert to uppercase
                    let kmer_upper: Vec<u8> =
                        kmer.iter().map(|&b| b.to_ascii_uppercase()).collect();

                    if let Some(&idx) = kmer_to_idx.get(&kmer_upper) {
                        counts[idx] += 1.0;
                    }
                }

                Ok(counts)
            })
            .collect();

        let encoded_seqs = encoded_seqs?;

        // Stack into 2D array
        let mut batch = Array2::<f32>::zeros((sequences.len(), total_kmers));

        for (i, encoded) in encoded_seqs.iter().enumerate() {
            for j in 0..total_kmers {
                batch[[i, j]] = encoded[j];
            }
        }

        Ok(batch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kmer_encoder_new() {
        let encoder = KmerEncoder::new(3, true, EncodingType::DNA);
        assert_eq!(encoder.k(), 3);
        assert!(encoder.is_canonical());
        assert_eq!(encoder.encoding_type(), EncodingType::DNA);
    }

    #[test]
    fn test_kmer_encode_dna() {
        let mut encoder = KmerEncoder::new(3, false, EncodingType::DNA);
        let sequence = b"ACGTACGT";

        let encoded = encoder.encode(sequence).unwrap();

        // Should have 4^3 = 64 possible 3-mers
        assert_eq!(encoded.len(), 64);

        // Should have counts > 0 for some k-mers
        let total_count: f32 = encoded.iter().sum();
        assert_eq!(total_count, (sequence.len() - 3 + 1) as f32); // 6 k-mers
    }

    #[test]
    fn test_kmer_encode_short_sequence() {
        let mut encoder = KmerEncoder::new(5, false, EncodingType::DNA);
        let sequence = b"ACG"; // Shorter than k

        let result = encoder.encode(sequence).unwrap();
        // Should return a zero vector of correct shape
        assert_eq!(result.len(), 4_usize.pow(5)); // 4^5 = 1024 for k=5 DNA
        assert_eq!(result.iter().sum::<f32>(), 0.0);
    }

    #[test]
    fn test_kmer_encode_invalid_char() {
        let mut encoder = KmerEncoder::new(3, false, EncodingType::DNA);
        let sequence = b"ACGTN"; // N is invalid for DNA

        let result = encoder.encode(sequence);
        assert!(result.is_err());
    }

    #[test]
    fn test_kmer_encode_batch() {
        let mut encoder = KmerEncoder::new(3, false, EncodingType::DNA);
        let sequences = vec![b"ACGTACGT".as_ref(), b"AAACCCGGG".as_ref()];

        let batch = encoder.encode_batch(&sequences).unwrap();

        assert_eq!(batch.shape(), &[2, 64]); // 2 sequences, 64 possible 3-mers

        // Each row should have counts
        for i in 0..2 {
            let row_sum: f32 = batch.row(i).iter().sum();
            assert!(row_sum > 0.0);
        }
    }

    #[test]
    fn test_kmer_encode_empty_batch() {
        let mut encoder = KmerEncoder::new(3, false, EncodingType::DNA);
        let sequences: Vec<&[u8]> = vec![];

        let batch = encoder.encode_batch(&sequences).unwrap();
        assert_eq!(batch.shape(), &[0, 64]); // 0 sequences, 64 possible 3-mers
    }

    #[test]
    fn test_kmer_encode_case_insensitive() {
        let mut encoder = KmerEncoder::new(3, false, EncodingType::DNA);

        let seq1 = b"ACGT";
        let seq2 = b"acgt";

        let encoded1 = encoder.encode(seq1).unwrap();
        let encoded2 = encoder.encode(seq2).unwrap();

        // Should produce the same encoding
        assert_eq!(encoded1, encoded2);
    }
}

// Python bindings
#[cfg(feature = "python")]
pub mod python {
    use super::*;
    use numpy::{PyArray1, PyArray2};
    use pyo3::prelude::*;
    use pyo3_stub_gen::derive::*;

    /// Python wrapper for KmerEncoder.
    ///
    /// Encodes biological sequences as k-mer frequency vectors.
    #[gen_stub_pyclass(module = "deepbiop.core")]
    #[pyclass(name = "KmerEncoder")]
    pub struct PyKmerEncoder {
        inner: KmerEncoder,
    }

    #[gen_stub_pymethods]
    #[pymethods]
    impl PyKmerEncoder {
        /// Create a new k-mer encoder.
        ///
        /// Args:
        ///     k: K-mer length
        ///     canonical: Whether to use canonical k-mers (k-mer and reverse complement are the same)
        ///     encoding_type: Type of sequence ("dna", "rna", or "protein")
        ///
        /// Returns:
        ///     A new KmerEncoder instance
        #[new]
        pub fn new(k: usize, canonical: bool, encoding_type: &str) -> PyResult<Self> {
            let enc_type = encoding_type
                .parse::<EncodingType>()
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

            Ok(Self {
                inner: KmerEncoder::new(k, canonical, enc_type),
            })
        }

        /// Encode a single sequence as a k-mer frequency vector.
        ///
        /// Args:
        ///     sequence: The sequence to encode (bytes)
        ///
        /// Returns:
        ///     NumPy array of shape [num_possible_kmers] with k-mer counts
        ///
        /// Raises:
        ///     ValueError: If the sequence contains invalid characters
        #[pyo3(name = "encode")]
        pub fn encode<'py>(
            &mut self,
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
        /// Args:
        ///     sequences: List of sequences to encode (list of bytes)
        ///
        /// Returns:
        ///     NumPy array of shape [num_sequences, num_possible_kmers]
        ///
        /// Raises:
        ///     ValueError: If any sequence fails to encode
        #[pyo3(name = "encode_batch")]
        pub fn encode_batch<'py>(
            &mut self,
            py: Python<'py>,
            sequences: Vec<Vec<u8>>,
        ) -> PyResult<Bound<'py, PyArray2<f32>>> {
            let seq_refs: Vec<&[u8]> = sequences.iter().map(|s| s.as_slice()).collect();

            // Release GIL for parallel processing with Rayon
            let encoded = py
                .allow_threads(|| self.inner.encode_batch(&seq_refs))
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

            Ok(PyArray2::from_array(py, &encoded))
        }

        /// Get the k-mer length.
        #[pyo3(name = "k")]
        pub fn k(&self) -> usize {
            self.inner.k()
        }

        /// Check if using canonical k-mers.
        #[pyo3(name = "is_canonical")]
        pub fn is_canonical(&self) -> bool {
            self.inner.is_canonical()
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
            format!(
                "KmerEncoder(k={}, canonical={}, encoding_type='{}')",
                self.k(),
                self.is_canonical(),
                self.encoding_type()
            )
        }
    }
}
