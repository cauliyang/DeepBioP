use ahash::HashSet;
use anyhow::{Ok, Result};
use bstr::BString;
use rayon::prelude::*;
use std::fs::File;
use std::io::{self, BufReader};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::thread;

use flate2::read::GzDecoder;
use noodles::fasta::record::{Definition, Record as FastaRecord, Sequence};
use noodles::{bgzf, fasta};

use crate::encode::RecordData;

/// Check if a file is gzip or bgzip compressed by examining its magic numbers.
///
/// This function reads the first few bytes of a file to detect if it's compressed,
/// without relying on file extensions.
///
/// # Arguments
///
/// * `path` - Path to the file to check
///
/// # Returns
///
/// A Result containing a tuple of two booleans (is_gzip, is_bgzip)
pub fn detect_compression<P: AsRef<Path>>(path: P) -> Result<(bool, bool)> {
    let mut file = File::open(path)?;
    let mut buffer = [0; 4];

    // Read first 4 bytes
    io::Read::read_exact(&mut file, &mut buffer)?;

    // Check gzip magic numbers (1f 8b)
    let is_gzip = buffer[0] == 0x1f && buffer[1] == 0x8b;

    // Check bgzip magic numbers (1f 8b 08 04)
    let is_bgzip = is_gzip && buffer[2] == 0x08 && buffer[3] == 0x04;

    Ok((is_gzip, is_bgzip))
}

/// Read FASTA records from a file, automatically detecting and handling compression.
///
/// This function takes a file path and reads FASTA records from it, automatically detecting
/// whether the file is uncompressed, gzip compressed (.gz), or bgzip compressed (.bgz).
/// It returns a vector of FASTA records.
///
/// # Arguments
///
/// * `file_path` - Path to the FASTA file, which may be compressed
///
/// # Returns
///
/// A Result containing a vector of FastaRecord on success, or an error on failure
///
/// # Example
///
/// ```no_run
/// use deepbiop_fa::io::read_noodel_records_from_fa_or_zip_fa;
/// use std::path::Path;
///
/// let records = read_noodel_records_from_fa_or_zip_fa("sequences.fa").unwrap();
/// let gzipped = read_noodel_records_from_fa_or_zip_fa("sequences.fa.gz").unwrap();
/// let bgzipped = read_noodel_records_from_fa_or_zip_fa("sequences.fa.bgz").unwrap();
/// ```
pub fn read_noodel_records_from_fa_or_zip_fa<P: AsRef<Path>>(
    file_path: P,
) -> Result<Vec<FastaRecord>> {
    let extension = file_path.as_ref().extension().unwrap();
    if extension == "bgz" {
        log::debug!("Reading from bgz file");
        read_noodle_records_from_bzip_fa(file_path)
    } else if extension == "gz" {
        log::debug!("Reading from gz file");
        read_noodle_records_from_gzip_fa(file_path)
    } else {
        log::debug!("Reading from fq file");
        read_noodle_records_from_fa(file_path)
    }
}

/// Read FASTA records from an uncompressed FASTA file.
///
/// This function reads FASTA records from an uncompressed file using the noodles library.
/// It processes the records in parallel for improved performance.
///
/// # Arguments
///
/// * `file_path` - Path to the uncompressed FASTA file
///
/// # Returns
///
/// A Result containing a vector of FastaRecord on success, or an error on failure
///
/// # Example
///
/// ```no_run
/// use deepbiop_fa::io::read_noodle_records_from_fa;
/// use std::path::Path;
///
/// let records = read_noodle_records_from_fa("sequences.fa").unwrap();
/// ```
pub fn read_noodle_records_from_fa<P: AsRef<Path>>(file_path: P) -> Result<Vec<FastaRecord>> {
    let mut reader = File::open(file_path)
        .map(BufReader::new)
        .map(fasta::Reader::new)?;
    let records: Result<Vec<FastaRecord>> = reader
        .records()
        .par_bridge()
        .map(|record| {
            let record = record?;
            Ok(record)
        })
        .collect();
    records
}

/// Read FASTA records from a gzip-compressed FASTA file.
///
/// This function reads FASTA records from a gzip-compressed file using the noodles library.
/// It processes the records in parallel for improved performance.
///
/// # Arguments
///
/// * `file_path` - Path to the gzip-compressed FASTA file
///
/// # Returns
///
/// A Result containing a vector of FastaRecord on success, or an error on failure
///
/// # Example
///
/// ```no_run
/// use deepbiop_fa::io::read_noodle_records_from_gzip_fa;
/// use std::path::Path;
///
/// let records = read_noodle_records_from_gzip_fa("sequences.fa.gz").unwrap();
/// ```
pub fn read_noodle_records_from_gzip_fa<P: AsRef<Path>>(file_path: P) -> Result<Vec<FastaRecord>> {
    let mut reader = File::open(file_path)
        .map(GzDecoder::new)
        .map(BufReader::new)
        .map(fasta::Reader::new)?;

    let records: Result<Vec<FastaRecord>> = reader
        .records()
        .par_bridge()
        .map(|record| {
            let record = record?;
            Ok(record)
        })
        .collect();
    records
}

/// Read FASTA records from a BGZF-compressed FASTA file.
///
/// This function reads FASTA records from a BGZF-compressed file using the noodles library.
/// It processes the records in parallel for improved performance.
///
/// # Arguments
///
/// * `file_path` - Path to the BGZF-compressed FASTA file
///
/// # Returns
///
/// A Result containing a vector of FastaRecord on success, or an error on failure
///
/// # Example
///
/// ```no_run
/// use deepbiop_fa::io::read_noodle_records_from_bzip_fa;
/// use std::path::Path;
///
/// let records = read_noodle_records_from_bzip_fa("sequences.fa.bgz").unwrap();
/// ```
pub fn read_noodle_records_from_bzip_fa<P: AsRef<Path>>(file_path: P) -> Result<Vec<FastaRecord>> {
    let decoder = bgzf::Reader::new(File::open(file_path)?);
    let mut reader = fasta::Reader::new(decoder);

    let records: Result<Vec<FastaRecord>> = reader
        .records()
        .par_bridge()
        .map(|record| {
            let record = record?;
            Ok(record)
        })
        .collect();
    records
}

/// Write FASTA records to a file or stdout.
///
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
    let encoder = bgzf::MultithreadedWriter::with_worker_count(worker_count, sink);

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
    let encoder = bgzf::MultithreadedWriter::with_worker_count(worker_count, sink);

    let mut writer = fasta::io::Writer::new(encoder);

    for record in records {
        writer.write_record(record)?;
    }
    Ok(())
}

pub fn convert_multiple_fas_to_one_zip_fa<P: AsRef<Path>>(
    paths: &[PathBuf],
    result_path: P,
    parallel: bool,
) -> Result<()> {
    let records = if parallel {
        paths
            .par_iter()
            .flat_map(|path| read_noodle_records_from_fa(path).unwrap())
            .collect::<Vec<FastaRecord>>()
    } else {
        paths
            .iter()
            .flat_map(|path| read_noodle_records_from_fa(path).unwrap())
            .collect::<Vec<FastaRecord>>()
    };
    write_bzip_fa_parallel_for_noodle_record(&records, result_path.as_ref().to_path_buf(), None)?;
    Ok(())
}

pub fn convert_multiple_zip_fas_to_one_zip_fa<P: AsRef<Path>>(
    paths: &[PathBuf],
    result_path: P,
    parallel: bool,
) -> Result<()> {
    let records = if parallel {
        paths
            .par_iter()
            .flat_map(|path| read_noodle_records_from_bzip_fa(path).unwrap())
            .collect::<Vec<FastaRecord>>()
    } else {
        paths
            .iter()
            .flat_map(|path| read_noodle_records_from_bzip_fa(path).unwrap())
            .collect::<Vec<FastaRecord>>()
    };
    write_bzip_fa_parallel_for_noodle_record(&records, result_path.as_ref().to_path_buf(), None)?;
    Ok(())
}

pub fn select_record_from_fa<P: AsRef<Path>>(
    fa: P,
    selected_records: &HashSet<BString>,
) -> Result<Vec<FastaRecord>> {
    let fa_records = read_noodle_records_from_fa(fa)?;

    Ok(fa_records
        .into_par_iter()
        .filter(|record| {
            let id: BString = record.name().to_vec().into();
            selected_records.contains(&id)
        })
        .collect())
}

pub fn select_record_from_fa_by_stream<P: AsRef<Path>>(
    fa: P,
    selected_records: &HashSet<BString>,
) -> Result<Vec<FastaRecord>> {
    let mut reader = File::open(fa).map(BufReader::new).map(fasta::Reader::new)?;

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
        let records = read_noodle_records_from_fa(test_file)?;

        // Check the number of records
        assert_eq!(records.len(), 14);
        Ok(())
    }
}
