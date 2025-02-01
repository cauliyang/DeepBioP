use anyhow::Result;
use log::info;
use noodles::fastq;
use std::{
    io::BufReader,
    path::{Path, PathBuf},
};

use super::record::RecordData;
use deepbiop_utils as utils;
use needletail::Sequence;

pub trait Encoder {
    type EncodeOutput;
    type RecordOutput;

    fn encode_multiple(&mut self, paths: &[PathBuf], parallel: bool) -> Self::EncodeOutput;
    fn encode<P: AsRef<Path>>(&mut self, path: P) -> Self::EncodeOutput;
    fn encode_record(&self, id: &[u8], seq: &[u8]) -> Self::RecordOutput;

    fn fetch_records<P: AsRef<Path>>(&mut self, path: P) -> Result<Vec<RecordData>> {
        info!("fetching records from {}", path.as_ref().display());
        let reader = utils::io::create_reader_for_compressed_file(path)?;
        let mut reader = fastq::Reader::new(BufReader::new(reader));

        let records: Vec<RecordData> = reader
            .records()
            .filter_map(|record| {
                let record = record.ok()?;

                let id = record.definition().name();
                let seq: &[u8] = record.sequence();
                let normalized_seq = seq.normalize(false);
                Some((id.to_vec(), normalized_seq.to_vec()).into())
            })
            .collect();
        info!("total records: {}", records.len());
        Ok(records)
    }
}
