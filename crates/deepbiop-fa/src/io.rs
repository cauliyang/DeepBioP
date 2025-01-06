use anyhow::Result;
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

pub fn read_noodel_records_from_fa_or_zip_fa<P: AsRef<Path>>(
    file_path: P,
) -> Result<Vec<FastaRecord>> {
    let extension = file_path.as_ref().extension().unwrap();
    if extension == "bgz" {
        log::info!("Reading from bgz file");
        read_noodle_records_from_bzip_fa(file_path)
    } else if extension == "gz" {
        log::info!("Reading from gz file");
        read_noodle_records_from_gzip_fa(file_path)
    } else {
        log::info!("Reading from fq file");
        read_noodle_records_from_fa(file_path)
    }
}

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

pub fn write_zip_fa_parallel(
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

pub fn write_fa_parallel_for_noodle_record(
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
    write_fa_parallel_for_noodle_record(&records, result_path.as_ref().to_path_buf(), None)?;
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
    write_fa_parallel_for_noodle_record(&records, result_path.as_ref().to_path_buf(), None)?;
    Ok(())
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
