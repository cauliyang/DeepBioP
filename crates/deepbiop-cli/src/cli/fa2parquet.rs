use anyhow::Result;
use clap::Parser;
use log::warn;

use std::path::PathBuf;

use super::set_up_threads;
use deepbiop_fa as fa;
use fa::encode::Encoder;

use deepbiop_utils as utils;

#[derive(Debug, Parser)]
pub struct FaToParquet {
    /// path to the fa file
    #[arg(value_name = "fa")]
    fa: PathBuf,

    /// if convert the fa file to parquet by chunk or not
    #[arg(long)]
    chunk: bool,

    /// chunk size
    #[arg(long, default_value = "1000000")]
    chunk_size: usize,

    /// result path
    #[arg(long, value_name = "result")]
    output: Option<PathBuf>,

    /// threads number
    #[arg(short, long, default_value = "2")]
    threads: Option<usize>,
}

impl FaToParquet {
    pub fn run(&self) -> Result<()> {
        set_up_threads(self.threads)?;
        let option = fa::encode::EncoderOptionBuilder::default()
            .bases(fa::encode::BASES.to_vec())
            .build()?;
        let mut fa_encoder = fa::encode::ParquetEncoderBuilder::default()
            .option(option)
            .build()?;

        if self.chunk {
            fa_encoder.encode_chunk(&self.fa, self.chunk_size, false)?;
            return Ok(());
        }

        let (record_batch, schema) = fa_encoder.encode(&self.fa)?;
        // result file is fq_path with .parquet extension
        let parquet_path = if let Some(path) = &self.output {
            if path.with_extension("parquet").exists() {
                warn!("{} already exists, overwriting", path.display());
            }
            path.with_extension("parquet")
        } else {
            self.fa.with_extension("parquet")
        };
        utils::io::write_parquet(parquet_path, record_batch, schema)?;

        Ok(())
    }
}
