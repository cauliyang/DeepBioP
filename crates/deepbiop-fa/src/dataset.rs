//! FASTA dataset implementation for streaming access to FASTA files.
//!
//! This module provides the `FastaDataset` struct that implements the `IterableDataset` trait
//! from `deepbiop-core`, enabling efficient streaming iteration over FASTA files without
//! loading the entire file into memory.
//!
//! # Features
//!
//! - Streaming iteration over FASTA records
//! - Automatic compression detection (plain text, gzip, bgzip)
//! - Memory-efficient processing of large files
//! - Integration with the `deepbiop-core` dataset trait system
//!
//! # Example
//!
//! ```no_run
//! use deepbiop_fa::dataset::FastaDataset;
//! use deepbiop_core::dataset::IterableDataset;
//!
//! let dataset = FastaDataset::new("data/sequences.fa").unwrap();
//!
//! for result in dataset.iter() {
//!     let record = result.unwrap();
//!     println!("ID: {}, Length: {}", record.id, record.sequence.len());
//! }
//! ```

use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

use anyhow::Result;
use noodles::fasta;

/// A dataset for streaming FASTA files.
///
/// `FastaDataset` provides memory-efficient iteration over FASTA files by streaming
/// records one at a time rather than loading the entire file into memory.
pub struct FastaDataset {
    file_path: String,
    records_count: Option<usize>,
}

impl FastaDataset {
    /// Creates a new `FastaDataset` from a file path.
    ///
    /// # Arguments
    ///
    /// * `file_path` - Path to the FASTA file (supports plain text, gzip, bgzip)
    ///
    /// # Returns
    ///
    /// A Result containing the FastaDataset or an error if the file cannot be opened.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use deepbiop_fa::dataset::FastaDataset;
    ///
    /// let dataset = FastaDataset::new("data/sequences.fa").unwrap();
    /// ```
    pub fn new<P: AsRef<Path>>(file_path: P) -> Result<Self> {
        let file_path_str = file_path.as_ref().to_string_lossy().to_string();

        // Check if file exists
        if !file_path.as_ref().exists() {
            return Err(anyhow::anyhow!("File does not exist: {}", file_path_str));
        }

        // Try to estimate record count (optional, for size_hint)
        let records_count = count_records_estimate(&file_path_str).ok();

        Ok(Self {
            file_path: file_path_str,
            records_count,
        })
    }

    /// Get total number of records in the dataset (if available).
    pub fn records_count(&self) -> Option<usize> {
        self.records_count
    }
}

// IterableDataset trait implementation for streaming access
impl deepbiop_core::dataset::IterableDataset for FastaDataset {
    fn iter(
        &self,
    ) -> Box<
        dyn Iterator<
                Item = deepbiop_core::dataset::DatasetResult<deepbiop_core::seq::SequenceRecord>,
            > + '_,
    > {
        Box::new(FastaStreamIterator::new(&self.file_path))
    }

    fn paths(&self) -> Vec<PathBuf> {
        vec![PathBuf::from(&self.file_path)]
    }

    fn size_hint(&self) -> Option<usize> {
        self.records_count
    }
}

/// Streaming iterator for IterableDataset trait implementation.
///
/// This provides true streaming access without loading the entire file.
struct FastaStreamIterator {
    reader: fasta::io::Reader<BufReader<Box<dyn Read>>>,
}

impl FastaStreamIterator {
    fn new(file_path: &str) -> Self {
        // Create reader with compression support
        let file_reader = deepbiop_utils::io::create_reader_for_compressed_file(file_path)
            .expect("Failed to create file reader");

        let buffered = BufReader::new(file_reader);
        let reader = fasta::io::Reader::new(buffered);

        Self { reader }
    }
}

impl Iterator for FastaStreamIterator {
    type Item = deepbiop_core::dataset::DatasetResult<deepbiop_core::seq::SequenceRecord>;

    fn next(&mut self) -> Option<Self::Item> {
        // Read next record
        match self.reader.records().next() {
            None => None, // EOF
            Some(Ok(record)) => {
                // Convert to SequenceRecord
                let id = String::from_utf8_lossy(record.name()).into_owned();
                let sequence = record.sequence().as_ref().to_vec();
                let quality_scores = None; // FASTA files don't have quality scores
                let description = record
                    .definition()
                    .description()
                    .map(|desc| desc.to_string());

                let seq_record = deepbiop_core::seq::SequenceRecord::new(
                    id,
                    sequence,
                    quality_scores,
                    description,
                );

                Some(Ok(seq_record))
            }
            Some(Err(e)) => {
                // Return error
                Some(Err(deepbiop_core::error::DPError::InvalidValue(format!(
                    "Failed to read FASTA record: {}",
                    e
                ))))
            }
        }
    }
}

/// Estimate the number of records in a FASTA file.
///
/// For large files, this samples the file to estimate the count.
/// For small files, it counts records exactly.
fn count_records_estimate(file_path: &str) -> Result<usize> {
    use std::fs::File;

    let file = File::open(file_path)?;
    let file_size = file.metadata()?.len() as usize;

    // If file is empty, return 0 immediately
    if file_size == 0 {
        return Ok(0);
    }

    // If file is small, count directly
    if file_size < 10_000_000 {
        // 10MB threshold
        return count_records_exact(file_path);
    }

    // For large files, sample to estimate
    let sample_size = std::cmp::min(file_size / 10, 5_000_000); // 5MB max, or 10% of file
    let mut reader = BufReader::with_capacity(65536, file);
    let mut sample_data = Vec::with_capacity(sample_size);

    use std::io::Read;
    reader
        .by_ref()
        .take(sample_size as u64)
        .read_to_end(&mut sample_data)?;

    // Count '>' characters in sample (FASTA record headers)
    let header_count = sample_data.iter().filter(|&&b| b == b'>').count();

    if header_count == 0 {
        // Fall back to exact counting since estimation is unreliable
        return count_records_exact(file_path);
    }

    // Estimate total records based on sample
    let estimated_records =
        (header_count as f64 * file_size as f64 / sample_data.len() as f64).ceil() as usize;

    Ok(estimated_records)
}

/// Count records exactly by reading through the entire file.
fn count_records_exact(file_path: &str) -> Result<usize> {
    let file_reader = deepbiop_utils::io::create_reader_for_compressed_file(file_path)?;
    let mut reader = fasta::io::Reader::new(BufReader::with_capacity(65536, file_reader));
    let count = reader.records().count();
    Ok(count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use deepbiop_core::dataset::IterableDataset;

    #[test]
    fn test_fasta_dataset_creation() {
        let dataset = FastaDataset::new("tests/data/test.fa").unwrap();
        assert!(dataset.size_hint().is_some());
    }

    #[test]
    fn test_fasta_dataset_iteration() {
        let dataset = FastaDataset::new("tests/data/test.fa").unwrap();
        let mut count = 0;

        for result in dataset.iter() {
            let record = result.unwrap();
            assert!(!record.id.is_empty());
            assert!(!record.sequence.is_empty());
            assert!(record.quality_scores.is_none()); // FASTA has no quality scores
            count += 1;
        }

        assert!(count > 0, "Dataset should contain records");
    }

    #[test]
    fn test_fasta_dataset_multiple_iterations() {
        let dataset = FastaDataset::new("tests/data/test.fa").unwrap();

        let count1: usize = dataset.iter().count();
        let count2: usize = dataset.iter().count();

        assert_eq!(count1, count2);
        assert!(count1 > 0);
    }

    #[test]
    fn test_nonexistent_file() {
        let result = FastaDataset::new("nonexistent.fa");
        assert!(result.is_err());
    }
}
