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

/// K-mer encoder for biological sequences.
///
/// Encodes sequences as dense k-mer count vectors over the full k-mer vocabulary.
/// With `canonical = true` (DNA/RNA only) a k-mer and its reverse complement share
/// one index, so the vocabulary shrinks to the number of canonical k-mers.
///
/// # Examples
///
/// ```
/// use deepbiop_core::kmer::encode::KmerEncoder;
/// use deepbiop_core::types::EncodingType;
///
/// let encoder = KmerEncoder::new(3, true, EncodingType::DNA).unwrap();
/// let encoded = encoder.encode(b"ACGTACGT").unwrap();
/// assert_eq!(encoded.len(), encoder.vocabulary_size());
/// ```
pub struct KmerEncoder {
    /// K-mer length
    k: usize,
    /// Use canonical k-mers (k-mer and reverse complement are the same)
    canonical: bool,
    /// Encoding type (DNA, RNA, or Protein)
    encoding_type: EncodingType,
    /// Every k-mer of the alphabet mapped to its output index (canonical k-mers
    /// and their reverse complements share an index).
    kmer_to_idx: HashMap<Vec<u8>, usize>,
    /// Number of distinct output indices.
    vocabulary_size: usize,
}

/// Largest dense vocabulary the encoder will materialize (16M entries).
pub const MAX_VOCABULARY_SIZE: usize = 1 << 24;

impl KmerEncoder {
    /// Create a new k-mer encoder.
    ///
    /// # Arguments
    ///
    /// * `k` - K-mer length (`1..`)
    /// * `canonical` - Fold each k-mer with its reverse complement (DNA/RNA only)
    /// * `encoding_type` - The type of sequence (DNA, RNA, or Protein)
    ///
    /// # Errors
    ///
    /// Returns an error if `k == 0`, if `alphabet_size^k` exceeds
    /// [`MAX_VOCABULARY_SIZE`], or if `canonical` is requested for protein.
    pub fn new(k: usize, canonical: bool, encoding_type: EncodingType) -> Result<Self> {
        if k == 0 {
            return Err(DPError::InvalidValue("k-mer length must be at least 1".into()).into());
        }
        if canonical && encoding_type == EncodingType::Protein {
            return Err(DPError::InvalidValue(
                "canonical k-mers are only defined for DNA/RNA".into(),
            )
            .into());
        }
        let alphabet = encoding_type.alphabet();
        let total = u32::try_from(k)
            .ok()
            .and_then(|k| alphabet.len().checked_pow(k))
            .filter(|&n| n <= MAX_VOCABULARY_SIZE)
            .ok_or_else(|| {
                DPError::InvalidValue(format!(
                    "k={k} yields more than {MAX_VOCABULARY_SIZE} {encoding_type:?} k-mers; choose a smaller k"
                ))
            })?;

        let mut kmer_to_idx = HashMap::with_capacity_and_hasher(total, Default::default());
        let mut next_idx = 0usize;
        let mut current = vec![alphabet[0]; k];
        // Iterate the vocabulary in lexicographic order via an odometer over alphabet positions.
        let mut digits = vec![0usize; k];
        loop {
            for (slot, &d) in current.iter_mut().zip(&digits) {
                *slot = alphabet[d];
            }
            let idx = if canonical {
                let rc = reverse_complement(&current, encoding_type);
                // The canonical representative is the lexicographically smaller strand; it
                // has already been assigned an index when it precedes `current`.
                match kmer_to_idx.get(&rc) {
                    Some(&i) if rc < current => i,
                    _ => {
                        next_idx += 1;
                        next_idx - 1
                    }
                }
            } else {
                next_idx += 1;
                next_idx - 1
            };
            kmer_to_idx.insert(current.clone(), idx);

            // Advance odometer.
            let mut pos = k;
            loop {
                if pos == 0 {
                    return Ok(Self {
                        k,
                        canonical,
                        encoding_type,
                        kmer_to_idx,
                        vocabulary_size: next_idx,
                    });
                }
                pos -= 1;
                digits[pos] += 1;
                if digits[pos] < alphabet.len() {
                    break;
                }
                digits[pos] = 0;
            }
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

    /// Length of the vectors produced by [`encode`](Self::encode).
    pub fn vocabulary_size(&self) -> usize {
        self.vocabulary_size
    }

    /// Encode a sequence as a k-mer count vector.
    ///
    /// # Errors
    ///
    /// Returns an error if the sequence is shorter than `k` or contains a
    /// character outside the alphabet.
    pub fn encode(&self, sequence: &[u8]) -> Result<Array1<f32>> {
        if sequence.len() < self.k {
            return Err(DPError::InvalidValue(format!(
                "sequence length {} is shorter than k={}",
                sequence.len(),
                self.k
            ))
            .into());
        }

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

        let mut counts = Array1::<f32>::zeros(self.vocabulary_size);
        let mut kmer = vec![0u8; self.k];
        for window in sequence.windows(self.k) {
            for (dst, &b) in kmer.iter_mut().zip(window) {
                *dst = b.to_ascii_uppercase();
            }
            if let Some(&idx) = self.kmer_to_idx.get(&kmer) {
                counts[idx] += 1.0;
            }
        }
        Ok(counts)
    }

    /// Encode multiple sequences in parallel as k-mer count vectors.
    ///
    /// # Errors
    ///
    /// Returns the first error produced by [`encode`](Self::encode).
    pub fn encode_batch(&self, sequences: &[&[u8]]) -> Result<Array2<f32>> {
        let mut batch = Array2::<f32>::zeros((sequences.len(), self.vocabulary_size));
        if sequences.is_empty() {
            return Ok(batch);
        }

        let rows: Vec<Array1<f32>> = sequences
            .par_iter()
            .map(|sequence| self.encode(sequence))
            .collect::<Result<_>>()?;
        for (mut dst, src) in batch.rows_mut().into_iter().zip(&rows) {
            dst.assign(src);
        }
        Ok(batch)
    }
}

fn reverse_complement(kmer: &[u8], encoding_type: EncodingType) -> Vec<u8> {
    kmer.iter()
        .rev()
        .map(|&b| match (b, encoding_type) {
            (b'A', EncodingType::RNA) => b'U',
            (b'U', EncodingType::RNA) => b'A',
            (b'A', _) => b'T',
            (b'T', _) => b'A',
            (b'C', _) => b'G',
            (b'G', _) => b'C',
            (other, _) => other,
        })
        .collect()
}

// Implement SequenceEncoder trait for KmerEncoder
impl crate::encoder::SequenceEncoder for KmerEncoder {
    type EncodeOutput = Array1<f32>;

    fn encode_sequence(&self, seq: &[u8], qual: Option<&[u8]>) -> Result<Self::EncodeOutput> {
        self.validate_input(seq, qual)?;
        self.encode(seq)
    }

    fn expected_output_size(&self, _seq_len: usize) -> usize {
        self.vocabulary_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kmer_encoder_new() {
        let encoder = KmerEncoder::new(3, true, EncodingType::DNA).unwrap();
        assert_eq!(encoder.k(), 3);
        assert!(encoder.is_canonical());
        assert_eq!(encoder.encoding_type(), EncodingType::DNA);
        // 64 3-mers fold to 32 canonical ones (no palindromic odd-length k-mers).
        assert_eq!(encoder.vocabulary_size(), 32);
    }

    #[test]
    fn test_kmer_encoder_rejects_bad_params() {
        assert!(KmerEncoder::new(0, false, EncodingType::DNA).is_err());
        assert!(KmerEncoder::new(40, false, EncodingType::DNA).is_err());
        assert!(KmerEncoder::new(3, true, EncodingType::Protein).is_err());
    }

    #[test]
    fn test_kmer_encode_dna() {
        let encoder = KmerEncoder::new(3, false, EncodingType::DNA).unwrap();
        let sequence = b"ACGTACGT";

        let encoded = encoder.encode(sequence).unwrap();

        assert_eq!(encoded.len(), 64);
        let total_count: f32 = encoded.iter().sum();
        assert_eq!(total_count, (sequence.len() - 3 + 1) as f32);
    }

    #[test]
    fn test_kmer_encode_canonical_folds_reverse_complement() {
        let plain = KmerEncoder::new(2, false, EncodingType::DNA).unwrap();
        let canonical = KmerEncoder::new(2, true, EncodingType::DNA).unwrap();

        // AC and its reverse complement GT land on different indices without folding...
        let ac = plain.encode(b"AC").unwrap();
        let gt = plain.encode(b"GT").unwrap();
        assert_ne!(ac, gt);
        // ...and on the same index with folding. AA/TT, AC/GT, AG/CT, AT, CA/TG, CC/GG, CG, GA/TC, TA -> 10.
        assert_eq!(canonical.vocabulary_size(), 10);
        assert_eq!(
            canonical.encode(b"AC").unwrap(),
            canonical.encode(b"GT").unwrap()
        );
        assert_eq!(
            canonical.encode(b"AA").unwrap(),
            canonical.encode(b"TT").unwrap()
        );
        assert_ne!(
            canonical.encode(b"AC").unwrap(),
            canonical.encode(b"CA").unwrap()
        );
    }

    #[test]
    fn test_kmer_encode_canonical_rna() {
        let canonical = KmerEncoder::new(2, true, EncodingType::RNA).unwrap();
        assert_eq!(
            canonical.encode(b"AC").unwrap(),
            canonical.encode(b"GU").unwrap()
        );
    }

    #[test]
    fn test_kmer_encode_short_sequence_errors() {
        let encoder = KmerEncoder::new(5, false, EncodingType::DNA).unwrap();
        assert!(encoder.encode(b"ACG").is_err());
    }

    #[test]
    fn test_kmer_encode_invalid_char() {
        let encoder = KmerEncoder::new(3, false, EncodingType::DNA).unwrap();
        assert!(encoder.encode(b"ACGTN").is_err());
    }

    #[test]
    fn test_kmer_encode_batch() {
        let encoder = KmerEncoder::new(3, false, EncodingType::DNA).unwrap();
        let sequences = vec![b"ACGTACGT".as_ref(), b"AAACCCGGG".as_ref()];

        let batch = encoder.encode_batch(&sequences).unwrap();

        assert_eq!(batch.shape(), &[2, 64]);
        assert_eq!(batch.row(0).sum(), 6.0);
        assert_eq!(batch.row(1).sum(), 7.0);
    }

    #[test]
    fn test_kmer_encode_empty_batch() {
        let encoder = KmerEncoder::new(3, false, EncodingType::DNA).unwrap();
        let sequences: Vec<&[u8]> = vec![];

        let batch = encoder.encode_batch(&sequences).unwrap();
        assert_eq!(batch.shape(), &[0, 64]);
    }

    #[test]
    fn test_kmer_encode_case_insensitive() {
        let encoder = KmerEncoder::new(3, false, EncodingType::DNA).unwrap();
        assert_eq!(
            encoder.encode(b"ACGT").unwrap(),
            encoder.encode(b"acgt").unwrap()
        );
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
    #[gen_stub_pyclass]
    #[pyclass(name = "KmerEncoder", module = "deepbiop.core")]
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
        ///
        /// Raises:
        ///     ValueError: If k is 0, the vocabulary would exceed 16M k-mers,
        ///         or canonical is requested for protein sequences
        #[new]
        pub fn new(k: usize, canonical: bool, encoding_type: &str) -> PyResult<Self> {
            let enc_type = encoding_type
                .parse::<EncodingType>()
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

            Ok(Self {
                inner: KmerEncoder::new(k, canonical, enc_type)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?,
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
                .detach(|| self.inner.encode_batch(&seq_refs))
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
