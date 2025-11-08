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
use log::debug;
#[cfg(feature = "cache")]
use log::info;
use serde::{Deserialize, Serialize};

use super::record::RecordDataBuilder;
#[cfg(feature = "cache")]
use deepbiop_utils::io::write_parquet;

use super::option::EncoderOption;

use anyhow::{Context, Result};
#[cfg(feature = "python")]
use pyo3::prelude::*;
use rayon::prelude::*;

#[cfg(feature = "python")]
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
#[cfg_attr(feature = "python", gen_stub_pyclass)]
#[cfg_attr(feature = "python", pyclass(module = "deepbiop.fa"))]
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

    fn generate_batches(
        &self,
        records: &[RecordData],
        schema: &Arc<Schema>,
    ) -> Result<Vec<RecordBatch>> {
        // Process smaller batches to avoid 2GB limit
        const BATCH_SIZE: usize = 10000; // Adjust this value based on your data size
        let all_batches: Vec<_> = records
            .par_chunks(BATCH_SIZE)
            .map(|chunk| {
                let _capacity = chunk.len();

                let mut id_builder = StringBuilder::new();
                let mut seq_builder = StringBuilder::new();

                for data in chunk {
                    let record = self
                        .encode_record(data.id.as_ref(), data.seq.as_ref())
                        .context(format!(
                            "encode fq read id {} error",
                            String::from_utf8_lossy(data.id.as_ref())
                        ))
                        .unwrap();
                    id_builder.append_value(record.id.to_string());
                    seq_builder.append_value(record.seq.to_string());
                }

                RecordBatch::try_new(
                    schema.clone(),
                    vec![
                        Arc::new(id_builder.finish()) as Arc<dyn Array>,
                        Arc::new(seq_builder.finish()) as Arc<dyn Array>,
                    ],
                )
                .unwrap()
            })
            .collect();
        debug!("all batches: {}", all_batches.len());
        Ok(all_batches)
    }

    #[cfg(feature = "cache")]
    fn generate_batch(&self, records: &[RecordData], schema: &Arc<Schema>) -> Result<RecordBatch> {
        let all_batches = self.generate_batches(records, schema)?;
        // Concatenate all batches
        arrow::compute::concat_batches(schema, &all_batches)
            .context("Failed to concatenate record batches")
    }

    #[cfg(feature = "cache")]
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
    type EncodeOutput = Result<(Vec<RecordBatch>, Arc<Schema>)>;
    type RecordOutput = Result<RecordData>;

    fn encode_multiple(&mut self, _paths: &[PathBuf], _parallel: bool) -> Self::EncodeOutput {
        todo!()
    }

    fn encode<P: AsRef<Path>>(&mut self, path: P) -> Self::EncodeOutput {
        // Define the schema of the data (one column of integers)
        let schema = self.generate_schema();
        let records = self.fetch_records(path)?;
        let record_batch = self.generate_batches(&records, &schema)?;
        Ok((record_batch, schema))
    }

    fn encode_record(&self, id: &[u8], seq: &[u8]) -> Self::RecordOutput {
        let result = RecordDataBuilder::default()
            .id(id.into())
            .seq(seq.into())
            .build()
            .context("Failed to build parquet data")?;
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use deepbiop_utils::io::write_parquet_for_batches;

    use crate::encode::option::EncoderOptionBuilder;

    use super::*;
    #[test]
    fn test_encode_fa_for_parquet() {
        let option = EncoderOptionBuilder::default().build().unwrap();
        let mut encoder = ParquetEncoderBuilder::default()
            .option(option)
            .build()
            .unwrap();

        let (record_batch, scheme) = encoder.encode("tests/data/test.fa").unwrap();
        write_parquet_for_batches("test.parquet", &record_batch, scheme).unwrap();
        // remove test.parquet
        std::fs::remove_file("test.parquet").unwrap();
    }
}
