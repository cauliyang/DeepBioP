use ahash::HashSet;
use anyhow::{Ok, Result};
use bstr::BString;
use rayon::prelude::*;
use std::fs::File;
use std::io::{self, BufReader};
use std::path::{Path, PathBuf};

use noodles::fasta::record::{Definition, Record as FastaRecord, Sequence};
use noodles::fastq::record::Record as FastqRecord;
use noodles::{fasta, fastq};

use crate::encode::RecordData;
use deepbiop_utils as utils;

pub fn read_noodle_records<P: AsRef<Path>>(file_path: P) -> Result<Vec<FastaRecord>> {
    let reader = utils::io::create_reader_for_compressed_file(&file_path)?;
    let mut reader = fasta::io::Reader::new(BufReader::new(reader));
    reader.records().map(|record| Ok(record?)).collect()
}

/// This function writes FASTA records to either a specified file or standard output.
/// Each record is written in FASTA format with a header line starting with '>' followed by the sequence.
///
/// # Arguments
///
/// * `records` - A slice of RecordData containing the sequences to write
/// * `file_path` - Optional path to output file. If None, writes to stdout
///
/// # Returns
///
/// A Result indicating success or failure of the write operation
///
/// # Example
///
/// ```no_run
/// use deepbiop_fa::io::write_fa;
/// use std::path::PathBuf;
/// use deepbiop_fa::encode::record::RecordData;
///
/// let records = vec![/* RecordData instances */];
/// let file_path = Some(PathBuf::from("output.fa"));
/// write_fa(&records, file_path).unwrap();
/// ```
pub fn write_fa(records: &[RecordData], file_path: Option<PathBuf>) -> Result<()> {
    let sink: Box<dyn io::Write> = if let Some(file) = file_path {
        Box::new(File::create(file)?)
    } else {
        Box::new(io::stdout().lock())
    };
    let mut writer = fasta::io::Writer::new(sink);

    for record in records {
        let record = fasta::Record::new(
            Definition::new(record.id.to_vec(), None),
            Sequence::from(record.seq.to_vec()),
        );
        writer.write_record(&record)?;
    }

    Ok(())
}

pub fn write_fa_for_noodle_record<P: AsRef<Path>>(data: &[fasta::Record], path: P) -> Result<()> {
    let file = std::fs::File::create(path.as_ref())?;
    let mut writer = fasta::io::Writer::new(file);
    for record in data {
        writer.write_record(record)?;
    }
    Ok(())
}

pub fn write_bzip_fa_parallel(
    records: &[RecordData],
    file_path: PathBuf,
    threads: Option<usize>,
) -> Result<()> {
    let encoder = utils::io::create_multithreaded_writer(file_path, threads)?;
    let mut writer = fasta::io::Writer::new(encoder);

    for record in records {
        let record = fasta::Record::new(
            Definition::new(record.id.to_vec(), None),
            Sequence::from(record.seq.to_vec()),
        );

        writer.write_record(&record)?;
    }
    Ok(())
}

pub fn write_bzip_fa_parallel_for_noodle_record(
    records: &[FastaRecord],
    file_path: PathBuf,
    threads: Option<usize>,
) -> Result<()> {
    let encoder = utils::io::create_multithreaded_writer(file_path, threads.or(Some(2)))?;
    let mut writer = fasta::io::Writer::new(encoder);

    for record in records {
        writer.write_record(record)?;
    }
    Ok(())
}

/// Converts FASTA records to FASTQ records with default quality scores.
///
/// Since FASTA files don't contain quality information, this function assigns
/// a default quality score (Phred+33 Q40, represented as '~') to all bases.
///
/// # Arguments
///
/// * `fa` - Path to the input FASTA file (supports plain, gzip, bgzip)
///
/// # Returns
///
/// A Result containing a Vec of FASTQ records
///
/// # Example
///
/// ```no_run
/// use deepbiop_fa::io::fasta_to_fastq;
/// use std::path::Path;
///
/// let fq_records = fasta_to_fastq(Path::new("input.fa")).unwrap();
/// ```
pub fn fasta_to_fastq<P: AsRef<Path>>(fa: P) -> Result<Vec<FastqRecord>> {
    let fa_records = read_noodle_records(&fa)?;
    log::info!("converting {} records", fa_records.len());

    let fq_records: Vec<FastqRecord> = fa_records
        .par_iter()
        .map(|fa_record| {
            let name = fa_record.name().to_vec();
            let sequence: Vec<u8> = fa_record.sequence().as_ref().to_vec();

            // Create default quality string with Q40 (Phred+33 = '~')
            // Q40 = 99.99% base call accuracy
            let quality = vec![b'~'; sequence.len()];

            let definition = fastq::record::Definition::new(name, "");
            FastqRecord::new(definition, sequence, quality)
        })
        .collect();

    Ok(fq_records)
}

/// Combines multiple FASTA files into a single bgzip-compressed FASTA file
///
/// # Arguments
///
/// * `paths` - A slice of paths to the input FASTA files
/// * `result_path` - Path where the combined bgzip FASTA file will be written
/// * `parallel` - Whether to process files in parallel using rayon
///
/// # Returns
///
/// Returns `Ok(())` if successful, or an error if file operations fail
///
/// # Example
///
/// ```no_run
/// use std::path::PathBuf;
/// use deepbiop_fa::io::convert_multiple_fas_to_one_bgzip_fa;
///
/// let input_files = vec![PathBuf::from("file1.fa"), PathBuf::from("file2.fa")];
/// let output = PathBuf::from("combined.fa.gz");
/// convert_multiple_fas_to_one_bgzip_fa(&input_files, output, true).unwrap();
/// ```
pub fn convert_multiple_fas_to_one_bgzip_fa<P: AsRef<Path>>(
    paths: &[PathBuf],
    result_path: P,
    parallel: bool,
) -> Result<()> {
    let records = if parallel {
        paths
            .par_iter()
            .flat_map(|path| read_noodle_records(path).unwrap())
            .collect::<Vec<FastaRecord>>()
    } else {
        paths
            .iter()
            .flat_map(|path| read_noodle_records(path).unwrap())
            .collect::<Vec<FastaRecord>>()
    };
    write_bzip_fa_parallel_for_noodle_record(&records, result_path.as_ref().to_path_buf(), None)?;
    Ok(())
}

/// Combines multiple FASTA files into a single bgzip-compressed FASTA file using streaming
///
/// This function uses a streaming approach to minimize memory usage. Instead of loading
/// all records into memory at once, it processes files one at a time in an iterator-like fashion.
///
/// # Arguments
///
/// * `paths` - A slice of paths to the input FASTA files
/// * `result_path` - Path where the combined bgzip FASTA file will be written
/// * `threads` - Optional number of threads for bgzip compression (default: 2)
///
/// # Returns
///
/// Returns `Ok(())` if successful, or an error if file operations fail
///
/// # Example
///
/// ```no_run
/// use std::path::PathBuf;
/// use deepbiop_fa::io::convert_multiple_fas_to_one_bgzip_fa_streaming;
///
/// let paths = vec![
///     PathBuf::from("file1.fa.gz"),
///     PathBuf::from("file2.fa.gz"),
/// ];
/// convert_multiple_fas_to_one_bgzip_fa_streaming(&paths, "output.fa.gz", None).unwrap();
/// ```
pub fn convert_multiple_fas_to_one_bgzip_fa_streaming<P: AsRef<Path>>(
    paths: &[PathBuf],
    result_path: P,
    threads: Option<usize>,
) -> Result<()> {
    let encoder = utils::io::create_multithreaded_writer(result_path, threads.or(Some(2)))?;
    let mut writer = fasta::io::Writer::new(encoder);

    // Process each file sequentially in a streaming fashion
    for path in paths {
        log::info!("Processing file: {:?}", path);
        let reader = utils::io::create_reader_for_compressed_file(path)?;
        let mut reader = fasta::io::Reader::new(BufReader::new(reader));

        // Stream records one at a time without loading all into memory
        for result in reader.records() {
            let record = result?;
            writer.write_record(&record)?;
        }
    }

    Ok(())
}

pub fn select_record_from_fa<P: AsRef<Path>>(
    fa: P,
    selected_records: &HashSet<BString>,
) -> Result<Vec<FastaRecord>> {
    let fa_records = read_noodle_records(fa)?;

    Ok(fa_records
        .into_par_iter()
        .filter(|record| {
            let id: BString = record.name().to_vec().into();
            selected_records.contains(&id)
        })
        .collect())
}

pub fn select_record_from_fa_by_random<P: AsRef<Path>>(
    fa: P,
    numbers: usize,
) -> Result<Vec<FastaRecord>> {
    let reader = utils::io::create_reader_for_compressed_file(fa)?;
    let mut reader = fasta::io::Reader::new(BufReader::new(reader));

    // Use reservoir sampling from utils
    let records_iter = reader.records().filter_map(|r| r.ok());
    Ok(utils::sampling::reservoir_sampling(records_iter, numbers))
}

pub fn select_record_from_fa_by_stream<P: AsRef<Path>>(
    fa: P,
    selected_records: &HashSet<BString>,
) -> Result<Vec<FastaRecord>> {
    let mut reader = File::open(fa)
        .map(BufReader::new)
        .map(fasta::io::Reader::new)?;

    reader
        .records()
        .par_bridge()
        .filter_map(|record| {
            let record = record.unwrap();
            let id: BString = record.name().to_vec().into();
            if selected_records.contains(&id) {
                Some(Ok(record))
            } else {
                None
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_noodle_records_from_fa() -> Result<()> {
        let test_file = "tests/data/test.fa";

        // Read records
        let records = read_noodle_records(test_file)?;

        // Check the number of records
        assert_eq!(records.len(), 14);
        Ok(())
    }

    #[test]
    fn test_fasta_to_fastq() -> Result<()> {
        let test_file = "tests/data/test.fa";

        // Convert FASTA to FASTQ
        let fq_records = fasta_to_fastq(test_file)?;

        // Check the number of records
        assert_eq!(fq_records.len(), 14);

        // Check first record has quality scores
        let first_record = &fq_records[0];
        let seq_len = first_record.sequence().len();
        let qual_len = first_record.quality_scores().len();
        assert_eq!(seq_len, qual_len);

        // Check all quality scores are Q40 ('~')
        let quality_bytes: &[u8] = first_record.quality_scores();
        for &q in quality_bytes {
            assert_eq!(q, b'~');
        }

        Ok(())
    }
}
