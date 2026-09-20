use anyhow::{anyhow, Result};
use ndarray::Array2;
use std::path::Path;

/// NumPy exporter for biological sequence data.
///
/// Exports sequence data to NumPy .npy format for direct loading into
/// Python ML frameworks. Supports both raw sequence data and encoded representations.
///
/// # Example
///
/// ```rust,no_run
/// use deepbiop_utils::export::numpy::NumpyExporter;
/// use std::path::Path;
///
/// let sequences = vec![b"ACGT".to_vec(), b"GGCC".to_vec()];
/// let exporter = NumpyExporter;
///
/// // Export as one-hot encoded
/// exporter.export_onehot(Path::new("sequences.npy"), &sequences, b"ACGT").unwrap();
/// ```
pub struct NumpyExporter;

impl NumpyExporter {
    /// Export sequences as one-hot encoded arrays.
    ///
    /// Each sequence is encoded as a 2D array where rows are positions
    /// and columns are bases (A, C, G, T by default).
    ///
    /// # Arguments
    ///
    /// * `path` - Output .npy file path
    /// * `sequences` - Slice of sequences to encode
    /// * `alphabet` - Alphabet for encoding (e.g., b"ACGT")
    ///
    /// # Returns
    ///
    /// Result indicating success or error
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use deepbiop_utils::export::numpy::NumpyExporter;
    /// # use std::path::Path;
    /// let seqs = vec![b"ACGT".to_vec()];
    /// NumpyExporter.export_onehot(Path::new("out.npy"), &seqs, b"ACGT").unwrap();
    /// ```
    pub fn export_onehot<P: AsRef<Path>>(
        &self,
        path: P,
        sequences: &[Vec<u8>],
        alphabet: &[u8],
    ) -> Result<()> {
        if sequences.is_empty() {
            return Err(anyhow!("No sequences to export"));
        }

        let max_len = sequences.iter().map(|s| s.len()).max().unwrap_or(0);
        let n_seqs = sequences.len();
        let n_bases = alphabet.len();

        // Create 3D array: (n_sequences, max_length, n_bases)
        let mut data = Array2::zeros((n_seqs * max_len, n_bases));

        for (seq_idx, seq) in sequences.iter().enumerate() {
            for (pos_idx, &base) in seq.iter().enumerate() {
                if let Some(base_idx) = alphabet
                    .iter()
                    .position(|&b| b == base || b == base.to_ascii_uppercase())
                {
                    let row = seq_idx * max_len + pos_idx;
                    data[[row, base_idx]] = 1.0;
                }
            }
        }

        // Reshape to 3D and write
        let data_3d = data
            .into_shape_with_order((n_seqs, max_len, n_bases))
            .map_err(|e| anyhow!("Failed to reshape array: {}", e))?;

        self.write_array(path, &data_3d)
    }

    /// Export sequences as integer encoded arrays.
    ///
    /// Each base is mapped to an integer based on the alphabet:
    /// A=0, C=1, G=2, T=3 (for alphabet "ACGT")
    ///
    /// # Arguments
    ///
    /// * `path` - Output .npy file path
    /// * `sequences` - Slice of sequences to encode
    /// * `alphabet` - Alphabet for encoding (e.g., b"ACGT")
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use deepbiop_utils::export::numpy::NumpyExporter;
    /// # use std::path::Path;
    /// let seqs = vec![b"ACGT".to_vec()];
    /// NumpyExporter.export_integer(Path::new("out.npy"), &seqs, b"ACGT").unwrap();
    /// ```
    pub fn export_integer<P: AsRef<Path>>(
        &self,
        path: P,
        sequences: &[Vec<u8>],
        alphabet: &[u8],
    ) -> Result<()> {
        if sequences.is_empty() {
            return Err(anyhow!("No sequences to export"));
        }

        let max_len = sequences.iter().map(|s| s.len()).max().unwrap_or(0);
        let n_seqs = sequences.len();

        // Create 2D array: (n_sequences, max_length)
        // Use 255 as padding value for sequences shorter than max_len
        let mut data = Array2::from_elem((n_seqs, max_len), 255u8);

        for (seq_idx, seq) in sequences.iter().enumerate() {
            for (pos_idx, &base) in seq.iter().enumerate() {
                if let Some(base_idx) = alphabet
                    .iter()
                    .position(|&b| b == base || b == base.to_ascii_uppercase())
                {
                    data[[seq_idx, pos_idx]] = base_idx as u8;
                }
            }
        }

        self.write_array(path, &data)
    }

    /// Export quality scores as a NumPy array.
    ///
    /// Quality scores are converted from ASCII to Phred scores.
    ///
    /// # Arguments
    ///
    /// * `path` - Output .npy file path
    /// * `qualities` - Slice of quality score strings
    /// * `offset` - Phred quality offset (33 for Phred+33, 64 for Phred+64)
    pub fn export_quality<P: AsRef<Path>>(
        &self,
        path: P,
        qualities: &[Vec<u8>],
        offset: u8,
    ) -> Result<()> {
        if qualities.is_empty() {
            return Err(anyhow!("No quality scores to export"));
        }

        let max_len = qualities.iter().map(|q| q.len()).max().unwrap_or(0);
        let n_seqs = qualities.len();

        // Create 2D array: (n_sequences, max_length)
        let mut data = Array2::zeros((n_seqs, max_len));

        for (seq_idx, qual) in qualities.iter().enumerate() {
            for (pos_idx, &q) in qual.iter().enumerate() {
                data[[seq_idx, pos_idx]] = (q - offset) as f32;
            }
        }

        self.write_array(path, &data)
    }

    /// Write an ndarray to a .npy file.
    ///
    /// Internal method for writing arrays in NumPy format.
    fn write_array<P, T, D>(
        &self,
        path: P,
        array: &ndarray::ArrayBase<ndarray::OwnedRepr<T>, D>,
    ) -> Result<()>
    where
        P: AsRef<Path>,
        T: ndarray_npy::WritableElement,
        D: ndarray::Dimension,
    {
        ndarray_npy::write_npy(path.as_ref(), array)?;
        Ok(())
    }
}

/// Helper function to export sequences and quality scores together.
///
/// Creates two .npy files: one for sequences and one for quality scores.
///
/// # Arguments
///
/// * `seq_path` - Output path for sequences .npy file
/// * `qual_path` - Output path for quality scores .npy file
/// * `sequences` - Slice of sequence data
/// * `qualities` - Slice of quality score data
/// * `alphabet` - Alphabet for encoding (e.g., b"ACGT")
/// * `quality_offset` - Phred quality offset (typically 33)
///
/// # Example
///
/// ```rust,no_run
/// use deepbiop_utils::export::numpy::export_fastq_to_numpy;
/// use std::path::Path;
///
/// let seqs = vec![b"ACGT".to_vec()];
/// let quals = vec![b"IIII".to_vec()];
///
/// export_fastq_to_numpy(
///     Path::new("seqs.npy"),
///     Path::new("quals.npy"),
///     &seqs,
///     &quals,
///     b"ACGT",
///     33
/// ).unwrap();
/// ```
pub fn export_fastq_to_numpy<P1: AsRef<Path>, P2: AsRef<Path>>(
    seq_path: P1,
    qual_path: P2,
    sequences: &[Vec<u8>],
    qualities: &[Vec<u8>],
    alphabet: &[u8],
    quality_offset: u8,
) -> Result<()> {
    if sequences.len() != qualities.len() {
        return Err(anyhow!(
            "Sequences and qualities must have same length: {} vs {}",
            sequences.len(),
            qualities.len()
        ));
    }

    let exporter = NumpyExporter;
    exporter.export_integer(seq_path, sequences, alphabet)?;
    exporter.export_quality(qual_path, qualities, quality_offset)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_export_onehot() -> Result<()> {
        let sequences = vec![b"ACGT".to_vec(), b"GGCC".to_vec()];
        let temp_file = NamedTempFile::new()?;

        let exporter = NumpyExporter;
        exporter.export_onehot(temp_file.path(), &sequences, b"ACGT")?;

        // Verify file was created
        let metadata = std::fs::metadata(temp_file.path())?;
        assert!(metadata.len() > 0);

        Ok(())
    }

    #[test]
    fn test_export_integer() -> Result<()> {
        let sequences = vec![b"ACGT".to_vec(), b"GG".to_vec()];
        let temp_file = NamedTempFile::new()?;

        let exporter = NumpyExporter;
        exporter.export_integer(temp_file.path(), &sequences, b"ACGT")?;

        let metadata = std::fs::metadata(temp_file.path())?;
        assert!(metadata.len() > 0);

        Ok(())
    }

    #[test]
    fn test_export_quality() -> Result<()> {
        let qualities = vec![b"IIII".to_vec(), b"!!!!".to_vec()];
        let temp_file = NamedTempFile::new()?;

        let exporter = NumpyExporter;
        exporter.export_quality(temp_file.path(), &qualities, 33)?;

        let metadata = std::fs::metadata(temp_file.path())?;
        assert!(metadata.len() > 0);

        Ok(())
    }

    #[test]
    fn test_export_fastq_to_numpy() -> Result<()> {
        let sequences = vec![b"ACGT".to_vec(), b"GGCC".to_vec()];
        let qualities = vec![b"IIII".to_vec(), b"!!!!".to_vec()];

        let seq_file = NamedTempFile::new()?;
        let qual_file = NamedTempFile::new()?;

        export_fastq_to_numpy(
            seq_file.path(),
            qual_file.path(),
            &sequences,
            &qualities,
            b"ACGT",
            33,
        )?;

        assert!(std::fs::metadata(seq_file.path())?.len() > 0);
        assert!(std::fs::metadata(qual_file.path())?.len() > 0);

        Ok(())
    }

    #[test]
    fn test_empty_sequences_error() {
        let sequences: Vec<Vec<u8>> = vec![];
        let temp_file = NamedTempFile::new().unwrap();

        let exporter = NumpyExporter;
        let result = exporter.export_onehot(temp_file.path(), &sequences, b"ACGT");

        assert!(result.is_err());
    }

    #[test]
    fn test_mismatched_lengths_error() {
        let sequences = vec![b"ACGT".to_vec()];
        let qualities = vec![b"IIII".to_vec(), b"!!!!".to_vec()];

        let seq_file = NamedTempFile::new().unwrap();
        let qual_file = NamedTempFile::new().unwrap();

        let result = export_fastq_to_numpy(
            seq_file.path(),
            qual_file.path(),
            &sequences,
            &qualities,
            b"ACGT",
            33,
        );

        assert!(result.is_err());
    }
}
