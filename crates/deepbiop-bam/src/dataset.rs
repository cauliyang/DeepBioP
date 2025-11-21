//! BAM dataset implementation for streaming access to BAM files.
//!
//! This module provides the `BamDataset` struct that implements the `IterableDataset` trait
//! from `deepbiop-core`, enabling efficient streaming iteration over BAM/SAM files without
//! loading the entire file into memory.
//!
//! # Features
//!
//! - Streaming iteration over BAM alignment records
//! - Multithreaded decompression support via bgzf
//! - Memory-efficient processing of large files
//! - Integration with the `deepbiop-core` dataset trait system
//! - Automatic conversion to SequenceRecord format
//!
//! # Example
//!
//! ```no_run
//! use deepbiop_bam::dataset::BamDataset;
//! use deepbiop_core::dataset::IterableDataset;
//!
//! let dataset = BamDataset::new("alignments.bam", None).unwrap();
//!
//! for result in dataset.iter() {
//!     let record = result.unwrap();
//!     println!("Read: {}, Length: {}", record.id, record.sequence.len());
//! }
//! ```

use std::fs::File;
use std::path::{Path, PathBuf};

use anyhow::Result;
use deepbiop_utils as utils;
use noodles::{bam, bgzf, sam};

/// Count total number of records in a BAM file.
///
/// This requires reading through the entire file once, but is necessary for
/// PyTorch DataLoader compatibility which requires __len__() support.
fn count_bam_records<P: AsRef<Path>>(file_path: P, threads: Option<usize>) -> Result<usize> {
    let file = File::open(file_path)?;
    let worker_count = utils::parallel::calculate_worker_count(threads);
    let decoder = bgzf::io::MultithreadedReader::with_worker_count(worker_count, file);
    let mut reader = bam::io::Reader::from(decoder);

    // Read and discard header
    let _ = reader.read_header()?;

    // Count records
    let count = reader.records().count();
    Ok(count)
}

/// A dataset for streaming BAM files.
///
/// `BamDataset` provides memory-efficient iteration over BAM files by streaming
/// alignment records one at a time rather than loading the entire file into memory.
pub struct BamDataset {
    file_path: String,
    threads: Option<usize>,
    records_count: Option<usize>,
}

impl BamDataset {
    /// Creates a new `BamDataset` from a file path.
    ///
    /// # Arguments
    ///
    /// * `file_path` - Path to the BAM file
    /// * `threads` - Optional number of threads for decompression (None = use all available)
    ///
    /// # Returns
    ///
    /// A Result containing the BamDataset or an error if the file cannot be opened.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use deepbiop_bam::dataset::BamDataset;
    ///
    /// // Use all available threads
    /// let dataset = BamDataset::new("alignments.bam", None).unwrap();
    ///
    /// // Use 4 threads
    /// let dataset = BamDataset::new("alignments.bam", Some(4)).unwrap();
    /// ```
    pub fn new<P: AsRef<Path>>(file_path: P, threads: Option<usize>) -> Result<Self> {
        let file_path_str = file_path.as_ref().to_string_lossy().to_string();

        // Check if file exists
        if !file_path.as_ref().exists() {
            return Err(anyhow::anyhow!("File does not exist: {}", file_path_str));
        }

        // Count records for __len__() support
        // While this reads through the file once, it's necessary for PyTorch DataLoader compatibility
        let records_count = Some(count_bam_records(&file_path, threads)?);

        Ok(Self {
            file_path: file_path_str,
            threads,
            records_count,
        })
    }

    /// Get total number of records in the dataset (if available).
    pub fn records_count(&self) -> Option<usize> {
        self.records_count
    }
}

// IterableDataset trait implementation for streaming access
impl deepbiop_core::dataset::IterableDataset for BamDataset {
    fn iter(
        &self,
    ) -> Box<
        dyn Iterator<
                Item = deepbiop_core::dataset::DatasetResult<deepbiop_core::seq::SequenceRecord>,
            > + '_,
    > {
        Box::new(BamStreamIterator::new(&self.file_path, self.threads))
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
struct BamStreamIterator {
    reader: bam::io::Reader<bgzf::io::MultithreadedReader<File>>,
    #[allow(dead_code)]
    header: sam::Header,
}

impl BamStreamIterator {
    fn new(file_path: &str, threads: Option<usize>) -> Self {
        // Open file
        let file = File::open(file_path).expect("Failed to open BAM file");

        // Calculate worker count for multithreaded decompression
        let worker_count = utils::parallel::calculate_worker_count(threads);

        // Create multithreaded bgzf reader
        let decoder = bgzf::io::MultithreadedReader::with_worker_count(worker_count, file);
        let mut reader = bam::io::Reader::from(decoder);

        // Read header
        let header = reader.read_header().expect("Failed to read BAM header");

        Self { reader, header }
    }
}

impl Iterator for BamStreamIterator {
    type Item = deepbiop_core::dataset::DatasetResult<deepbiop_core::seq::SequenceRecord>;

    fn next(&mut self) -> Option<Self::Item> {
        // Read next record
        match self.reader.records().next() {
            None => None, // EOF
            Some(Ok(record)) => {
                // Convert BAM record to SequenceRecord
                let id = record
                    .name()
                    .map(|name| String::from_utf8_lossy(name.as_ref()).into_owned())
                    .unwrap_or_else(|| "unknown".to_string());

                let sequence = record.sequence().as_ref().to_vec();
                let quality_scores = Some(record.quality_scores().as_ref().to_vec());

                // BAM records don't have a description field, but we could include
                // some alignment information if needed
                let description = None;

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
                    "Failed to read BAM record: {}",
                    e
                ))))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use deepbiop_core::dataset::IterableDataset;

    #[test]
    fn test_bam_dataset_creation() {
        let dataset = BamDataset::new("tests/data/test_chimric_reads.bam", None).unwrap();
        // BAM files don't have quick record counting, so size_hint may be None
        let _ = dataset.size_hint();
    }

    #[test]
    fn test_bam_dataset_iteration() {
        let dataset = BamDataset::new("tests/data/test_chimric_reads.bam", None).unwrap();
        let mut count = 0;

        for result in dataset.iter() {
            let record = result.unwrap();
            assert!(!record.id.is_empty());
            assert!(!record.sequence.is_empty());
            assert!(record.quality_scores.is_some()); // BAM has quality scores
            count += 1;
        }

        assert!(count > 0, "Dataset should contain records");
    }

    #[test]
    fn test_bam_dataset_multiple_iterations() {
        let dataset = BamDataset::new("tests/data/test_chimric_reads.bam", None).unwrap();

        let count1: usize = dataset.iter().count();
        let count2: usize = dataset.iter().count();

        assert_eq!(count1, count2);
        assert!(count1 > 0);
    }

    #[test]
    fn test_bam_dataset_with_threads() {
        let dataset = BamDataset::new("tests/data/test_chimric_reads.bam", Some(4)).unwrap();
        let count: usize = dataset.iter().count();
        assert!(count > 0);
    }

    #[test]
    fn test_nonexistent_file() {
        let result = BamDataset::new("nonexistent.bam", None);
        assert!(result.is_err());
    }
}
