use ahash::HashSet;
use anyhow::{Ok, Result};
use bstr::BString;
use rand::{rng, Rng};
use rayon::prelude::*;
use std::fs::File;
use std::io::{self, BufReader};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::thread;

use noodles::fasta::record::{Definition, Record as FastaRecord, Sequence};
use noodles::{bgzf, fasta};

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
    let worker_count = NonZeroUsize::new(threads.unwrap_or(1))
        .map(|count| count.min(thread::available_parallelism().unwrap()))
        .unwrap();

    let sink = File::create(file_path)?;
    let encoder = bgzf::io::MultithreadedWriter::with_worker_count(worker_count, sink);

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
    let worker_count = NonZeroUsize::new(threads.unwrap_or(2))
        .map(|count| count.min(thread::available_parallelism().unwrap()))
        .unwrap();

    let sink = File::create(file_path)?;
    let encoder = bgzf::io::MultithreadedWriter::with_worker_count(worker_count, sink);

    let mut writer = fasta::io::Writer::new(encoder);

    for record in records {
        writer.write_record(record)?;
    }
    Ok(())
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

pub fn select_record_from_fq_by_random<P: AsRef<Path>>(
    fa: P,
    numbers: usize,
) -> Result<Vec<FastaRecord>> {
    let reader = utils::io::create_reader_for_compressed_file(fa)?;
    let mut reader = fasta::io::Reader::new(BufReader::new(reader));

    // Use reservoir sampling algorithm to randomly select records
    let mut rng = rng();
    let mut selected_records = Vec::with_capacity(numbers);
    let mut count = 0;

    let records_iter = reader.records().filter_map(|r| r.ok());
    let mut records_iter = records_iter.peekable();

    // Fill reservoir with first k elements
    while selected_records.len() < numbers && records_iter.peek().is_some() {
        if let Some(record) = records_iter.next() {
            selected_records.push(record);
            count += 1;
        }
    }

    // Process remaining elements with reservoir sampling
    for record in records_iter {
        count += 1;
        let j = rng.random_range(0..count);
        if j < numbers {
            selected_records[j] = record;
        }
    }

    if count < numbers {
        selected_records.truncate(count);
    }
    Ok(selected_records)
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
}
