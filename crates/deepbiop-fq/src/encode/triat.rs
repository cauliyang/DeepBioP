use anyhow::Result;
use log::info;
use needletail::Sequence;
use rayon::prelude::*;
use std::path::{Path, PathBuf};

use crate::io;
use crate::types::Element;

use super::RecordData;

pub trait Encoder {
    type EncodeOutput;
    type RecordOutput;

    fn encode_multiple(&mut self, paths: &[PathBuf], parallel: bool) -> Self::EncodeOutput;

    fn encode<P: AsRef<Path>>(&mut self, path: P) -> Self::EncodeOutput;

    fn encode_record(&self, id: &[u8], seq: &[u8], qual: &[u8]) -> Self::RecordOutput;

    fn fetch_records<P: AsRef<Path>>(&mut self, path: P) -> Result<Vec<RecordData>> {
        info!("fetching records from {}", path.as_ref().display());
        let _records = io::read_noodle_records(path)?;

        let records: Vec<RecordData> = _records
            .into_par_iter()
            .filter_map(|record| {
                let id = record.definition().name();
                let seq = record.sequence();
                let normalized_seq = seq.normalize(false);
                let qual = record.quality_scores();
                let seq_len = normalized_seq.len();
                let qual_len = qual.len();

                if seq_len != qual_len {
                    // NOTE: it seems like log mes does not work well with rayon paralllel iterator  <02-26-24, Yangyang Li>
                    // warn!(
                    //     "record: id {} seq_len != qual_len",
                    //     String::from_utf8_lossy(id)
                    // );
                    return None;
                }

                Some((id.to_vec(), seq.to_vec(), qual.to_vec()).into())
            })
            .collect();

        info!("total records: {}", records.len());
        Ok(records)
    }

    fn encode_qual(&self, qual: &[u8], qual_offset: u8) -> Vec<Element> {
        // input is quality of fastq
        // 1. convert the quality to a score
        // 2. return the score
        let encoded_qual: Vec<u8> = qual
            .par_iter()
            .map(|&q| {
                // Convert ASCII to Phred score for Phred+33 encoding
                q - qual_offset
            })
            .collect();
        encoded_qual.into_par_iter().map(|x| x as Element).collect()
    }

    fn encoder_seq<'a>(&self, seq: &'a [u8]) -> Result<&'a [u8]> {
        Ok(seq)
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_parse_target_from_id() {
        // Test case 1: Valid input
        struct TestEncoder;
        impl Encoder for TestEncoder {
            type RecordOutput = Result<RecordData>;
            type EncodeOutput = Result<Vec<RecordData>>;

            fn encode_multiple(
                &mut self,
                _paths: &[PathBuf],
                _parallel: bool,
            ) -> Self::EncodeOutput {
                Ok(Vec::new())
            }
            fn encode<P: AsRef<Path>>(&mut self, _path: P) -> Self::EncodeOutput {
                Ok(Vec::new())
            }
            fn encode_record(&self, _id: &[u8], _seq: &[u8], _qual: &[u8]) -> Self::RecordOutput {
                Ok(RecordData::default())
            }
        }
    }
}
