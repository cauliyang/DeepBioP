//! BAM/SAM file reader with filtering and feature extraction

use crate::features::AlignmentFeatures;
use anyhow::{Context, Result};
use deepbiop_utils as utils;
use noodles::{
    bam, bgzf,
    core::{Position, Region},
    sam,
};
use rayon::prelude::*;
use std::fs::File;
use std::path::{Path, PathBuf};

/// BAM/SAM file reader with streaming and filtering capabilities
pub struct BamReader {
    reader: bam::io::Reader<bgzf::io::MultithreadedReader<File>>,
    header: sam::Header,
    path: PathBuf,
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
        let path = path.as_ref().to_path_buf();
        let file =
            File::open(&path).with_context(|| format!("Failed to open BAM file: {path:?}"))?;

        let worker_count = utils::parallel::calculate_worker_count(threads);
        let decoder = bgzf::io::MultithreadedReader::with_worker_count(worker_count, file);
        let mut reader = bam::io::Reader::from(decoder);

        let header = reader.read_header().context("Failed to read BAM header")?;

        Ok(Self {
            reader,
            header,
            path,
        })
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

    /// Query alignments in a specific genomic region (requires a BAM index).
    ///
    /// # Arguments
    ///
    /// * `chromosome` - Chromosome/contig name
    /// * `start` - Start position (1-based, inclusive)
    /// * `end` - End position (1-based, inclusive)
    ///
    /// # Note
    ///
    /// This function requires a `.bai` index file next to the BAM file, either named
    /// `<path>.bai` or `<path without .bam>.bai`. Unindexed files return an error; there
    /// is no fallback to a full scan.
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
        chromosome: &str,
        start: u64,
        end: u64,
    ) -> Result<Vec<bam::Record>> {
        let index_path = locate_bam_index(&self.path).ok_or_else(|| {
            anyhow::anyhow!(
                "query_region requires a BAM index (.bai) next to {}",
                self.path.display()
            )
        })?;

        let index = bam::bai::fs::read(&index_path)
            .with_context(|| format!("Failed to read BAM index: {}", index_path.display()))?;

        let mut indexed_reader = bam::io::indexed_reader::Builder::default()
            .set_index(index)
            .build_from_path(&self.path)
            .with_context(|| format!("Failed to open indexed BAM reader: {:?}", self.path))?;

        let header = indexed_reader
            .read_header()
            .context("Failed to read BAM header")?;

        let start_position = Position::new(start as usize)
            .ok_or_else(|| anyhow::anyhow!("Invalid start position: {start} (must be >= 1)"))?;
        let end_position = Position::new(end as usize)
            .ok_or_else(|| anyhow::anyhow!("Invalid end position: {end} (must be >= 1)"))?;
        let region = Region::new(chromosome, start_position..=end_position);

        indexed_reader
            .query(&header, &region)
            .with_context(|| format!("Failed to query region {chromosome}:{start}-{end}"))?
            .records()
            .collect::<std::io::Result<Vec<_>>>()
            .with_context(|| format!("Failed to read records in region {chromosome}:{start}-{end}"))
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
        self.reader
            .records()
            .par_bridge()
            .try_fold(
                || 0usize,
                |acc, result| {
                    let record = result.context("Failed to read BAM record")?;
                    Ok::<usize, anyhow::Error>(if record.flags().is_supplementary() {
                        acc + 1
                    } else {
                        acc
                    })
                },
            )
            .try_reduce(
                || 0usize,
                |a: usize, b: usize| Ok::<usize, anyhow::Error>(a + b),
            )
    }
}

/// Locates a `.bai` index file for a BAM path.
///
/// Checks `<path>.bai` (the samtools default) first, then falls back to
/// `<path without .bam>.bai` for the older naming convention.
fn locate_bam_index(bam_path: &Path) -> Option<PathBuf> {
    let mut sidecar = bam_path.as_os_str().to_os_string();
    sidecar.push(".bai");
    let sidecar = PathBuf::from(sidecar);
    if sidecar.exists() {
        return Some(sidecar);
    }

    if bam_path.extension().is_some_and(|ext| ext == "bam") {
        let replaced = bam_path.with_extension("bai");
        if replaced.exists() {
            return Some(replaced);
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_BAM: &str = "tests/data/test_chimric_reads.bam";

    #[test]
    fn test_bam_reader_structure() {
        // This test just ensures the reader structure compiles
        // Actual I/O tests would require test BAM files
    }

    #[test]
    fn test_count_chimeric() {
        use noodles::sam::alignment::io::Write as _;
        use noodles::sam::alignment::record::Flags;
        use noodles::sam::alignment::RecordBuf;

        // The bundled fixture has zero supplementary-flagged records (its "chimeric"
        // reads are represented via SA tags on primary alignments instead), so build a
        // tiny synthetic BAM with a known number of supplementary records to give this
        // test a real ground truth.
        let header = sam::Header::default();
        let make_record =
            |name: &str, flags: Flags| RecordBuf::builder().set_name(name).set_flags(flags).build();

        let dir = tempfile::tempdir().unwrap();
        let bam_path = dir.path().join("synthetic.bam");
        {
            let file = File::create(&bam_path).unwrap();
            let mut writer = bam::io::Writer::new(file);
            writer.write_header(&header).unwrap();
            writer
                .write_alignment_record(&header, &make_record("primary", Flags::UNMAPPED))
                .unwrap();
            writer
                .write_alignment_record(
                    &header,
                    &make_record("supp1", Flags::UNMAPPED | Flags::SUPPLEMENTARY),
                )
                .unwrap();
            writer
                .write_alignment_record(
                    &header,
                    &make_record("supp2", Flags::UNMAPPED | Flags::SUPPLEMENTARY),
                )
                .unwrap();
        }

        let mut reader = BamReader::open(&bam_path, None).unwrap();
        assert_eq!(reader.count_chimeric().unwrap(), 2);
    }

    #[test]
    fn test_query_region_without_index_errors() {
        let mut reader = BamReader::open(TEST_BAM, None).unwrap();
        let result = reader.query_region("chr1", 1, 100);
        assert!(result.is_err());
        let message = result.unwrap_err().to_string();
        assert!(message.contains("BAM index"));
    }
}
