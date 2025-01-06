use anyhow::Result;
use rayon::prelude::*;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use flate2::read::GzDecoder;
use noodles::fasta::record::Record as FastaRecord;
use noodles::{bgzf, fasta};

pub fn read_noodel_records_from_fq_or_zip_fa<P: AsRef<Path>>(
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_read_noodle_records_from_fa() -> Result<()> {
        let test_file = "tests/data/test.fa";

        // Read records
        let records = read_noodle_records_from_fa(test_file)?;

        // Check the number of records
        assert_eq!(records.len(), 14);
        // Cleanup
        fs::remove_file(test_file)?;
        Ok(())
    }
}
