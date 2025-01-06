use anyhow::Result;
use log::info;
use needletail::Sequence;
use rayon::prelude::*;
use std::path::{Path, PathBuf};

use super::record::RecordData;

pub trait Encoder {
    type TargetOutput;
    type EncodeOutput;
    type RecordOutput;

    fn encode_multiple(&mut self, paths: &[PathBuf], parallel: bool) -> Self::EncodeOutput;
    fn encode<P: AsRef<Path>>(&mut self, path: P) -> Self::EncodeOutput;
    fn encode_record(&self, id: &[u8], seq: &[u8]) -> Self::RecordOutput;

    fn fetch_records<P: AsRef<Path>>(&mut self, path: P) -> Result<Vec<RecordData>> {
        info!("fetching records from {}", path.as_ref().display());
        let _records = crate::io::read_noodel_records_from_fa_or_zip_fa(path)?;

        let records: Vec<RecordData> = _records
            .into_par_iter()
            .filter_map(|record| {
                let id = record.definition().name();
                let seq = record.sequence().as_ref();
                let normalized_seq = seq.normalize(false);
                Some((id.to_vec(), normalized_seq.to_vec()).into())
            })
            .collect();
        info!("total records: {}", records.len());
        Ok(records)
    }
}
