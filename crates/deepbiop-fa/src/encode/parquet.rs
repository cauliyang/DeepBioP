use crate::encode::record::RecordData;
use std::{
    fmt::Display,
    path::{Path, PathBuf},
    sync::Arc,
};

use crate::encode::traits::Encoder;
use arrow::array::{Array, RecordBatch, StringBuilder};
use arrow::datatypes::{DataType, Field, Schema};

use derive_builder::Builder;
use log::info;
use serde::{Deserialize, Serialize};

use super::record::RecordDataBuilder;
use deepbiop_utils::io::write_parquet;

use super::option::EncoderOption;

use anyhow::{Context, Result};
use pyo3::prelude::*;
use rayon::prelude::*;

use pyo3_stub_gen::derive::*;

/// An encoder for converting FASTA records to Parquet format.
///
/// This struct provides functionality to encode FASTA sequence data into Parquet files,
/// which are an efficient columnar storage format.
///
/// # Fields
///
/// * `option` - Configuration options for the encoder, including which bases to consider
///
/// # Example
///
/// ```
/// use deepbiop_fa::encode::{option::EncoderOption, parquet::ParquetEncoder};
///
/// let options = EncoderOption::default();
/// let encoder = ParquetEncoder::new(options);
/// ```
#[gen_stub_pyclass]
#[pyclass(module = "deepbiop.fa")]
#[derive(Debug, Builder, Default, Clone, Serialize, Deserialize)]
pub struct ParquetEncoder {
    pub option: EncoderOption,
}

impl ParquetEncoder {
    pub fn new(option: EncoderOption) -> Self {
        Self { option }
    }

    fn generate_schema(&self) -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("seq", DataType::Utf8, false),
        ]))
    }

    fn generate_batch(&self, records: &[RecordData], schema: &Arc<Schema>) -> Result<RecordBatch> {
        let data: Vec<RecordData> = records
            .into_par_iter()
            .filter_map(|data| {
                let id = data.id.as_ref();
                let seq = data.seq.as_ref();
                match self.encode_record(id, seq).context(format!(
                    "encode fq read id {} error",
                    String::from_utf8_lossy(id)
                )) {
                    Ok(result) => Some(result),
                    Err(_e) => None,
                }
            })
            .collect();

        // Create builders for each field
        let mut id_builder = StringBuilder::new();
        let mut seq_builder = StringBuilder::new();

        // Populate builders
        data.into_iter().for_each(|parquet_record| {
            id_builder.append_value(parquet_record.id.to_string());
            seq_builder.append_value(parquet_record.seq.to_string());
        });

        // Build arrays
        let id_array = Arc::new(id_builder.finish());
        let seq_array = Arc::new(seq_builder.finish());

        // Create a RecordBatch
        let record_batch = RecordBatch::try_new(
            schema.clone(),
            vec![id_array as Arc<dyn Array>, seq_array as Arc<dyn Array>],
        )?;
        Ok(record_batch)
    }

    pub fn encode_chunk<P: AsRef<Path>>(
        &mut self,
        path: P,
        chunk_size: usize,
        parallel: bool,
    ) -> Result<()> {
        let schema = self.generate_schema();
        let records = self.fetch_records(&path)?;
        info!("Encoding records with chunk size {} ", chunk_size);

        // create a folder for the chunk parquet files
        let file_name = path.as_ref().file_name().unwrap().to_str().unwrap();
        let chunks_folder = path
            .as_ref()
            .parent()
            .unwrap()
            .join(format!("{}_{}", file_name, "chunks"))
            .to_path_buf();
        // create the folder
        std::fs::create_dir_all(&chunks_folder).context("Failed to create folder for chunks")?;

        if parallel {
            records
                // .chunks(chunk_size)
                .par_chunks(chunk_size)
                .enumerate()
                .for_each(|(idx, record)| {
                    let record_batch = self
                        .generate_batch(record, &schema)
                        .context(format!("Failed to generate record batch for chunk {}", idx))
                        .unwrap();
                    let parquet_path = chunks_folder.join(format!("{}_{}.parquet", file_name, idx));
                    write_parquet(parquet_path, record_batch, schema.clone())
                        .context(format!("Failed to write parquet file for chunk {}", idx))
                        .unwrap();
                });
        } else {
            records
                .chunks(chunk_size)
                .enumerate()
                .for_each(|(idx, record)| {
                    let record_batch = self
                        .generate_batch(record, &schema)
                        .context(format!("Failed to generate record batch for chunk {}", idx))
                        .unwrap();
                    let parquet_path = chunks_folder.join(format!("{}_{}.parquet", file_name, idx));
                    write_parquet(parquet_path, record_batch, schema.clone())
                        .context(format!("Failed to write parquet file for chunk {}", idx))
                        .unwrap();
                });
        }

        Ok(())
    }
}

impl Display for ParquetEncoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "FaEncoder {{ option: {} }}", self.option)
    }
}

impl Encoder for ParquetEncoder {
    type RecordOutput = Result<RecordData>;
    type EncodeOutput = Result<(RecordBatch, Arc<Schema>)>;

    fn encode_record(&self, id: &[u8], seq: &[u8]) -> Self::RecordOutput {
        let result = RecordDataBuilder::default()
            .id(id.into())
            .seq(seq.into())
            .build()
            .context("Failed to build parquet data")?;
        Ok(result)
    }

    fn encode<P: AsRef<Path>>(&mut self, path: P) -> Self::EncodeOutput {
        // Define the schema of the data (one column of integers)
        let schema = self.generate_schema();
        let records = self.fetch_records(path)?;
        let record_batch = self.generate_batch(&records, &schema)?;
        Ok((record_batch, schema))
    }

    fn encode_multiple(&mut self, _paths: &[PathBuf], _parallel: bool) -> Self::EncodeOutput {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use crate::encode::option::EncoderOptionBuilder;

    use super::*;
    #[test]
    fn test_encode_fq_for_parquet() {
        let option = EncoderOptionBuilder::default().build().unwrap();
        let mut encoder = ParquetEncoderBuilder::default()
            .option(option)
            .build()
            .unwrap();

        let (record_batch, scheme) = encoder.encode("tests/data/test.fa").unwrap();
        write_parquet("test.parquet", record_batch, scheme).unwrap();
        // remove test.parquet
        std::fs::remove_file("test.parquet").unwrap();
    }
}
