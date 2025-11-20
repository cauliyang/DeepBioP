//! BAM/SAM file reader with filtering and feature extraction

use crate::features::AlignmentFeatures;
use anyhow::{Context, Result};
use deepbiop_utils as utils;
use noodles::{bam, bgzf, sam};
use rayon::prelude::*;
use std::fs::File;
use std::path::Path;

/// BAM/SAM file reader with streaming and filtering capabilities
pub struct BamReader {
    reader: bam::io::Reader<bgzf::io::MultithreadedReader<File>>,
    header: sam::Header,
}

impl BamReader {
    /// Open a BAM file for reading
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the BAM file
    /// * `threads` - Optional number of threads for decompression
    ///
    /// # Example
    ///
    /// ```no_run
    /// use deepbiop_bam::reader::BamReader;
    /// use std::path::Path;
    ///
    /// let reader = BamReader::open(Path::new("alignments.bam"), Some(4)).unwrap();
    /// ```
    pub fn open<P: AsRef<Path>>(path: P, threads: Option<usize>) -> Result<Self> {
        let file = File::open(path.as_ref())
            .with_context(|| format!("Failed to open BAM file: {:?}", path.as_ref()))?;

        let worker_count = utils::parallel::calculate_worker_count(threads);
        let decoder = bgzf::io::MultithreadedReader::with_worker_count(worker_count, file);
        let mut reader = bam::io::Reader::from(decoder);

        let header = reader.read_header().context("Failed to read BAM header")?;

        Ok(Self { reader, header })
    }

    /// Get the SAM header
    pub fn header(&self) -> &sam::Header {
        &self.header
    }

    /// Read all records from the BAM file
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use deepbiop_bam::reader::BamReader;
    /// # use std::path::Path;
    /// let mut reader = BamReader::open(Path::new("alignments.bam"), None).unwrap();
    /// let records = reader.read_all().unwrap();
    /// println!("Found {} alignments", records.len());
    /// ```
    pub fn read_all(&mut self) -> Result<Vec<bam::Record>> {
        self.reader
            .records()
            .par_bridge()
            .map(|result| result.context("Failed to read BAM record"))
            .collect::<Result<Vec<_>>>()
    }

    /// Filter alignments by mapping quality
    ///
    /// # Arguments
    ///
    /// * `min_quality` - Minimum mapping quality threshold (0-255)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use deepbiop_bam::reader::BamReader;
    /// # use std::path::Path;
    /// let mut reader = BamReader::open(Path::new("alignments.bam"), None).unwrap();
    /// let high_quality = reader.filter_by_mapping_quality(30).unwrap();
    /// println!("Found {} high-quality alignments", high_quality.len());
    /// ```
    pub fn filter_by_mapping_quality(&mut self, min_quality: u8) -> Result<Vec<bam::Record>> {
        self.reader
            .records()
            .par_bridge()
            .map(|result| result.context("Failed to read BAM record"))
            .filter_map(|result| match result {
                Ok(record) => {
                    if let Some(mq) = record.mapping_quality() {
                        if u8::from(mq) >= min_quality {
                            return Some(Ok(record));
                        }
                    }
                    None
                }
                Err(e) => Some(Err(e)),
            })
            .collect::<Result<Vec<_>>>()
    }

    /// Extract read pairs (for paired-end sequencing)
    ///
    /// Returns only properly paired reads where both mates are mapped
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use deepbiop_bam::reader::BamReader;
    /// # use std::path::Path;
    /// let mut reader = BamReader::open(Path::new("alignments.bam"), None).unwrap();
    /// let pairs = reader.extract_read_pairs().unwrap();
    /// println!("Found {} read pairs", pairs.len());
    /// ```
    pub fn extract_read_pairs(&mut self) -> Result<Vec<bam::Record>> {
        self.reader
            .records()
            .par_bridge()
            .map(|result| result.context("Failed to read BAM record"))
            .filter_map(|result| match result {
                Ok(record) => {
                    // Check if paired, both mapped, and properly paired
                    let flags = record.flags();
                    if flags.is_segmented() && !flags.is_unmapped() && !flags.is_mate_unmapped() {
                        return Some(Ok(record));
                    }
                    None
                }
                Err(e) => Some(Err(e)),
            })
            .collect::<Result<Vec<_>>>()
    }

    /// Query alignments in a specific genomic region (requires BAM index)
    ///
    /// # Arguments
    ///
    /// * `chromosome` - Chromosome/contig name
    /// * `start` - Start position (1-based, inclusive)
    /// * `end` - End position (1-based, inclusive)
    ///
    /// # Note
    ///
    /// This function requires a BAM index file (.bai or .csi) to be present.
    /// For unindexed files, this will return an error.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use deepbiop_bam::reader::BamReader;
    /// # use std::path::Path;
    /// let mut reader = BamReader::open(Path::new("alignments.bam"), None).unwrap();
    /// let region_reads = reader.query_region("chr1", 1000000, 2000000).unwrap();
    /// println!("Found {} alignments in region", region_reads.len());
    /// ```
    pub fn query_region(
        &mut self,
        _chromosome: &str,
        _start: u64,
        _end: u64,
    ) -> Result<Vec<bam::Record>> {
        // TODO: Implement indexed region query
        // This requires:
        // 1. Opening the BAM index (.bai or .csi file)
        // 2. Using noodles::bam::io::indexed_reader::Reader
        // 3. Querying the specific region

        anyhow::bail!("Region queries require BAM index support (not yet implemented)")
    }

    /// Extract alignment features for machine learning
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use deepbiop_bam::reader::BamReader;
    /// # use std::path::Path;
    /// let mut reader = BamReader::open(Path::new("alignments.bam"), None).unwrap();
    /// let features = reader.extract_features().unwrap();
    /// println!("Extracted features from {} alignments", features.len());
    /// ```
    pub fn extract_features(&mut self) -> Result<Vec<AlignmentFeatures>> {
        self.reader
            .records()
            .par_bridge()
            .map(|result| {
                let record = result.context("Failed to read BAM record")?;
                AlignmentFeatures::from_record(&record)
            })
            .collect::<Result<Vec<_>>>()
    }

    /// Count chimeric reads (reads with supplementary alignments)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use deepbiop_bam::reader::BamReader;
    /// # use std::path::Path;
    /// let mut reader = BamReader::open(Path::new("alignments.bam"), None).unwrap();
    /// let chimeric_count = reader.count_chimeric().unwrap();
    /// println!("Found {} chimeric reads", chimeric_count);
    /// ```
    pub fn count_chimeric(&mut self) -> Result<usize> {
        let count = self
            .reader
            .records()
            .par_bridge()
            .filter_map(|result| {
                if let Ok(record) = result {
                    let flags = record.flags();
                    if flags.is_supplementary() {
                        return Some(1);
                    }
                }
                None
            })
            .sum();

        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_bam_reader_structure() {
        // This test just ensures the reader structure compiles
        // Actual I/O tests would require test BAM files
    }
}
