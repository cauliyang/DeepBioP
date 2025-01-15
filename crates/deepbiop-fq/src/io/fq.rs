use std::fs::File;
use std::io::BufReader;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::{io, thread};

use anyhow::Result;
use arrow::datatypes::ToByteSlice;
use noodles::fastq::{self as fastq, record::Definition};

use ahash::HashSet;
use bstr::BString;
use noodles::bgzf;
use noodles::fasta;
use noodles::fastq::record::Record as FastqRecord;
use rayon::prelude::*;

use flate2::read::GzDecoder;

use crate::encode::RecordData;

pub fn read_noodel_records_from_fq_or_zip_fq<P: AsRef<Path>>(
    file_path: P,
) -> Result<Vec<FastqRecord>> {
    let extension = file_path.as_ref().extension().unwrap();
    if extension == "bgz" {
        log::info!("Reading from bgz file");
        read_noodle_records_from_bzip_fq(file_path)
    } else if extension == "gz" {
        log::info!("Reading from gz file");
        read_noodle_records_from_gzip_fq(file_path)
    } else {
        log::info!("Reading from fq file");
        read_noodle_records_from_fq(file_path)
    }
}

pub fn read_noodle_records_from_fq<P: AsRef<Path>>(file_path: P) -> Result<Vec<FastqRecord>> {
    let mut reader = File::open(file_path)
        .map(BufReader::new)
        .map(fastq::Reader::new)?;

    let records: Result<Vec<FastqRecord>> = reader
        .records()
        .par_bridge()
        .map(|record| {
            let record = record?;
            Ok(record)
        })
        .collect();
    records
}

pub fn read_noodle_records_from_gzip_fq<P: AsRef<Path>>(file_path: P) -> Result<Vec<FastqRecord>> {
    let mut reader = File::open(file_path)
        .map(GzDecoder::new)
        .map(BufReader::new)
        .map(fastq::Reader::new)?;

    let records: Result<Vec<FastqRecord>> = reader
        .records()
        .par_bridge()
        .map(|record| {
            let record = record?;
            Ok(record)
        })
        .collect();
    records
}

pub fn read_noodle_records_from_bzip_fq<P: AsRef<Path>>(file_path: P) -> Result<Vec<FastqRecord>> {
    let decoder = bgzf::Reader::new(File::open(file_path)?);
    let mut reader = fastq::Reader::new(decoder);

    let records: Result<Vec<FastqRecord>> = reader
        .records()
        .par_bridge()
        .map(|record| {
            let record = record?;
            Ok(record)
        })
        .collect();
    records
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

pub fn write_bzip_fq_parallel(
    records: &[RecordData],
    file_path: PathBuf,
    threads: Option<usize>,
) -> Result<()> {
    let worker_count = NonZeroUsize::new(threads.unwrap_or(1))
        .map(|count| count.min(thread::available_parallelism().unwrap()))
        .unwrap();

    let sink = File::create(file_path)?;
    let encoder = bgzf::MultithreadedWriter::with_worker_count(worker_count, sink);

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

pub fn write_bzip_fq_parallel_for_noodle_record(
    records: &[FastqRecord],
    file_path: PathBuf,
    threads: Option<usize>,
) -> Result<()> {
    let worker_count = NonZeroUsize::new(threads.unwrap_or(2))
        .map(|count| count.min(thread::available_parallelism().unwrap()))
        .unwrap();

    let sink = File::create(file_path)?;
    let encoder = bgzf::MultithreadedWriter::with_worker_count(worker_count, sink);

    let mut writer = fastq::io::Writer::new(encoder);

    for record in records {
        writer.write_record(record)?;
    }
    Ok(())
}

pub fn convert_multiple_bzip_fqs_to_one_bzip_fq<P: AsRef<Path>>(
    paths: &[PathBuf],
    result_path: P,
    parallel: bool,
) -> Result<()> {
    let records = if parallel {
        paths
            .par_iter()
            .flat_map(|path| read_noodle_records_from_bzip_fq(path).unwrap())
            .collect::<Vec<FastqRecord>>()
    } else {
        paths
            .iter()
            .flat_map(|path| read_noodle_records_from_bzip_fq(path).unwrap())
            .collect::<Vec<FastqRecord>>()
    };
    write_bzip_fq_parallel_for_noodle_record(&records, result_path.as_ref().to_path_buf(), None)?;
    Ok(())
}

pub fn convert_multiple_fqs_to_one_bzip_fq<P: AsRef<Path>>(
    paths: &[PathBuf],
    result_path: P,
    parallel: bool,
) -> Result<()> {
    let records = if parallel {
        paths
            .par_iter()
            .flat_map(|path| read_noodle_records_from_fq(path).unwrap())
            .collect::<Vec<FastqRecord>>()
    } else {
        paths
            .iter()
            .flat_map(|path| read_noodle_records_from_fq(path).unwrap())
            .collect::<Vec<FastqRecord>>()
    };
    write_bzip_fq_parallel_for_noodle_record(&records, result_path.as_ref().to_path_buf(), None)?;
    Ok(())
}

pub fn fastq_to_fasta<P: AsRef<Path>>(fq: P) -> Result<Vec<fasta::Record>> {
    let fq_records = read_noodel_records_from_fq_or_zip_fq(&fq)?;
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

pub fn select_record_from_fq<P: AsRef<Path>>(
    fq: P,
    selected_records: &HashSet<BString>,
) -> Result<Vec<FastqRecord>> {
    let fq_records = read_noodel_records_from_fq_or_zip_fq(&fq)?;
    log::info!("load {} reads in total", fq_records.len());

    Ok(fq_records
        .into_par_iter()
        .filter(|record| {
            let id: BString = record.name().into();
            selected_records.contains(&id)
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    use tempfile::NamedTempFile;

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
        write_bzip_fq_parallel(&records, file_path, None).unwrap();

        let decoder = bgzf::Reader::new(file.reopen().unwrap());
        let mut reader = fastq::Reader::new(decoder);

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
