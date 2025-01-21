use anyhow::Result;
use clap::Parser;
use log::warn;

use std::path::PathBuf;

use super::set_up_threads;
use deepbiop_fq as fq;
use fq::encode::Encoder;

use deepbiop_core::default;
use deepbiop_utils as utils;

#[derive(Debug, Parser)]
pub struct FqToParquet {
    /// path to the fq file
    #[arg(value_name = "fq")]
    fq: PathBuf,

    /// if convert the fq file to parquet by chunk or not
    #[arg(long)]
    chunk: bool,

    /// chunk size
    #[arg(long, default_value = "1000000")]
    chunk_size: usize,

    /// result path
    #[arg(long, value_name = "output")]
    output: Option<PathBuf>,

    /// threads number
    #[arg(short, long, default_value = "2")]
    threads: Option<usize>,
}

impl FqToParquet {
    pub fn run(&self) -> Result<()> {
        set_up_threads(self.threads)?;
        let option = fq::encode::EncoderOptionBuilder::default()
            .bases(default::BASES.to_vec())
            .build()?;
        let mut fq_encoder = fq::encode::ParquetEncoderBuilder::default()
            .option(option)
            .build()?;

        if self.chunk {
            fq_encoder.encode_chunk(&self.fq, self.chunk_size, false)?;
            return Ok(());
        }

        let (record_batch, schema) = fq_encoder.encode(&self.fq)?;
        // result file is fq_path with .parquet extension
        let parquet_path = if let Some(path) = &self.output {
            if path.with_extension("parquet").exists() {
                warn!("{} already exists, overwriting", path.display());
            }
            path.with_extension("parquet")
        } else {
            self.fq.with_extension("parquet")
        };
        utils::io::write_parquet(parquet_path, record_batch, schema)?;

        Ok(())
    }
}
