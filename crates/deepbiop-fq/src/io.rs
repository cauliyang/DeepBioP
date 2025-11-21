use std::fs::File;
use std::io;
use std::io::BufReader;
use std::path::{Path, PathBuf};

use anyhow::Result;
use arrow::datatypes::ToByteSlice;
use noodles::fastq::{self as fastq, record::Definition};

use ahash::HashSet;
use bstr::BString;
use noodles::fasta;
use noodles::fastq::record::Record as FastqRecord;
use rayon::prelude::*;

use crate::encode::RecordData;
use deepbiop_utils as utils;

pub fn read_noodle_records<P: AsRef<Path>>(file_path: P) -> Result<Vec<FastqRecord>> {
    let reader = utils::io::create_reader_for_compressed_file(&file_path)?;
    let mut reader = fastq::io::Reader::new(BufReader::new(reader));
    reader.records().map(|record| Ok(record?)).collect()
}

pub fn write_fq(records: &[RecordData], file_path: Option<PathBuf>) -> Result<()> {
    let sink: Box<dyn io::Write> = if let Some(file) = file_path {
        Box::new(File::create(file)?)
    } else {
        Box::new(io::stdout().lock())
    };
    let mut writer = fastq::io::Writer::new(sink);

    for record in records {
        let qual_str = record.qual.to_string();

        let record = fastq::Record::new(
            Definition::new(record.id.to_vec(), ""),
            record.seq.to_vec(),
            qual_str,
        );
        writer.write_record(&record)?;
    }

    Ok(())
}

pub fn write_fq_for_noodle_record<P: AsRef<Path>>(data: &[fastq::Record], path: P) -> Result<()> {
    let file = std::fs::File::create(path.as_ref())?;
    let mut writer = fastq::io::Writer::new(file);
    for record in data {
        writer.write_record(record)?;
    }
    Ok(())
}

pub fn write_bgzip_fq_parallel(
    records: &[RecordData],
    file_path: PathBuf,
    threads: Option<usize>,
) -> Result<()> {
    let encoder = utils::io::create_multithreaded_writer(file_path, threads)?;
    let mut writer = fastq::io::Writer::new(encoder);

    for record in records {
        let record = fastq::Record::new(
            Definition::new(record.id.to_vec(), ""),
            record.seq.to_vec(),
            record.qual.to_vec(),
        );
        writer.write_record(&record)?;
    }
    Ok(())
}

pub fn write_bgzip_fq_parallel_for_noodle_record(
    records: &[FastqRecord],
    file_path: PathBuf,
    threads: Option<usize>,
) -> Result<()> {
    let encoder = utils::io::create_multithreaded_writer(file_path, threads.or(Some(2)))?;
    let mut writer = fastq::io::Writer::new(encoder);

    for record in records {
        writer.write_record(record)?;
    }
    Ok(())
}

/// Combines multiple FASTQ files into a single bgzip-compressed FASTQ file
///
/// # Arguments
///
/// * `paths` - A slice of paths to the input FASTQ files
/// * `result_path` - Path where the combined bgzip FASTQ file will be written
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
/// use deepbiop_fq::io::convert_multiple_fqs_to_one_bgzip_fq;
///
/// let input_files = vec![PathBuf::from("file1.fq"), PathBuf::from("file2.fq")];
/// let output = PathBuf::from("combined.fq.gz");
/// convert_multiple_fqs_to_one_bgzip_fq(&input_files, output, true).unwrap();
/// ```
pub fn convert_multiple_fqs_to_one_bgzip_fq<P: AsRef<Path>>(
    paths: &[PathBuf],
    result_path: P,
    parallel: bool,
) -> Result<()> {
    let records = if parallel {
        paths
            .par_iter()
            .flat_map(|path| read_noodle_records(path).unwrap())
            .collect::<Vec<FastqRecord>>()
    } else {
        paths
            .iter()
            .flat_map(|path| read_noodle_records(path).unwrap())
            .collect::<Vec<FastqRecord>>()
    };
    write_bgzip_fq_parallel_for_noodle_record(&records, result_path.as_ref().to_path_buf(), None)?;
    Ok(())
}

/// Combines multiple FASTQ files into a single bgzip-compressed FASTQ file using streaming
///
/// This function uses a streaming approach to minimize memory usage. Instead of loading
/// all records into memory at once, it processes files one at a time in an iterator-like fashion.
///
/// # Arguments
///
/// * `paths` - A slice of paths to the input FASTQ files
/// * `result_path` - Path where the combined bgzip FASTQ file will be written
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
/// use deepbiop_fq::io::convert_multiple_fqs_to_one_bgzip_fq_streaming;
///
/// let paths = vec![
///     PathBuf::from("file1.fq.gz"),
///     PathBuf::from("file2.fq.gz"),
/// ];
/// convert_multiple_fqs_to_one_bgzip_fq_streaming(&paths, "output.fq.gz", None).unwrap();
/// ```
pub fn convert_multiple_fqs_to_one_bgzip_fq_streaming<P: AsRef<Path>>(
    paths: &[PathBuf],
    result_path: P,
    threads: Option<usize>,
) -> Result<()> {
    let encoder = utils::io::create_multithreaded_writer(result_path, threads.or(Some(2)))?;
    let mut writer = fastq::io::Writer::new(encoder);

    // Process each file sequentially in a streaming fashion
    for path in paths {
        log::info!("Processing file: {:?}", path);
        let reader = utils::io::create_reader_for_compressed_file(path)?;
        let mut reader = fastq::io::Reader::new(BufReader::new(reader));

        // Stream records one at a time without loading all into memory
        for result in reader.records() {
            let record = result?;
            writer.write_record(&record)?;
        }
    }

    Ok(())
}

/// Converts a FASTQ file to FASTA records
///
/// # Arguments
///
/// * `fq` - Path to the input FASTQ file
///
/// # Returns
///
/// Returns a Result containing a Vec of FASTA records if successful, or an error if file operations fail
///
/// # Example
///
/// ```no_run
/// use std::path::PathBuf;
/// use deepbiop_fq::io::fastq_to_fasta;
///
/// let input = PathBuf::from("input.fq");
/// let fasta_records = fastq_to_fasta(input).unwrap();
/// ```
pub fn fastq_to_fasta<P: AsRef<Path>>(fq: P) -> Result<Vec<fasta::Record>> {
    let fq_records = read_noodle_records(&fq)?;
    log::info!("converting {} records", fq_records.len());

    let fa_records: Vec<fasta::Record> = fq_records
        .par_iter()
        .map(|fq_record| {
            let definition =
                fasta::record::Definition::new(fq_record.definition().name().to_byte_slice(), None);
            let seq = fasta::record::Sequence::from(fq_record.sequence().to_vec());
            fasta::Record::new(definition, seq)
        })
        .collect();

    Ok(fa_records)
}

pub fn select_record_from_fq_by_random<P: AsRef<Path>>(
    fq: P,
    numbers: usize,
) -> Result<Vec<FastqRecord>> {
    let reader = utils::io::create_reader_for_compressed_file(fq)?;
    let mut reader = fastq::io::Reader::new(BufReader::new(reader));

    // Use reservoir sampling from utils
    let records_iter = reader.records().filter_map(|r| r.ok());
    Ok(utils::sampling::reservoir_sampling(records_iter, numbers))
}

pub fn select_record_from_fq<P: AsRef<Path>>(
    fq: P,
    selected_records: &HashSet<BString>,
) -> Result<Vec<FastqRecord>> {
    let reader = utils::io::create_reader_for_compressed_file(fq)?;
    let mut reader = fastq::io::Reader::new(BufReader::new(reader));

    reader
        .records()
        .filter_map(|record| {
            let record = record.unwrap();
            let id: BString = record.definition().name().into();
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
    use noodles::bgzf;

    use tempfile::NamedTempFile;

    #[test]

    fn test_read_records() -> Result<()> {
        let fq = "./tests/data/one_record.fq";
        let gzip_fq = "./tests/data/one_record.fq.gz";
        let bgzip_fq = "./tests/data/one_recordbgz.fq.gz";

        let records = read_noodle_records(fq)?;
        assert_eq!(records.len(), 1);

        let records = read_noodle_records(gzip_fq)?;
        assert_eq!(records.len(), 1);

        let records = read_noodle_records(bgzip_fq)?;
        assert_eq!(records.len(), 1);

        Ok(())
    }

    #[test]
    fn test_write_fq_with_file_path() {
        let records = vec![
            RecordData {
                id: b"1".into(),
                seq: b"ATCG".into(),
                qual: b"HHHH".into(),
            },
            RecordData {
                id: b"2".into(),
                seq: b"GCTA".into(),
                qual: b"MMMM".into(),
            },
        ];
        let file = NamedTempFile::new().unwrap();
        let file_path = Some(file.path().to_path_buf());

        write_fq(&records, file_path).unwrap();

        let contents = std::fs::read_to_string(file.path()).unwrap();
        assert_eq!(contents, "@1\nATCG\n+\nHHHH\n@2\nGCTA\n+\nMMMM\n");
    }

    #[test]
    fn test_write_fq_parallel() {
        // Create some test data
        let records = vec![
            RecordData {
                id: b"record1".into(),
                seq: b"ATCG".into(),
                qual: b"IIII".into(),
            },
            RecordData {
                id: b"record2".into(),
                seq: b"GCTA".into(),
                qual: b"EEEE".into(),
            },
        ];

        // Create a temporary file to write the records to
        let file = NamedTempFile::new().unwrap();
        let file_path = file.path().to_path_buf();

        // Call the function being tested
        write_bgzip_fq_parallel(&records, file_path, None).unwrap();

        let decoder = bgzf::io::Reader::new(file.reopen().unwrap());
        let mut reader = fastq::io::Reader::new(decoder);

        let actual_result: Vec<RecordData> = reader
            .records()
            .par_bridge()
            .map(|record| {
                let record = record.unwrap();
                let id = record.definition().name();
                let seq = record.sequence();
                let qual = record.quality_scores();
                RecordData {
                    id: id.into(),
                    seq: seq.into(),
                    qual: qual.into(),
                }
            })
            .collect();

        actual_result.iter().zip(records.iter()).for_each(|(a, b)| {
            assert_eq!(a.id, b.id);
            assert_eq!(a.seq, b.seq);
            assert_eq!(a.qual, b.qual);
        });
    }
}
