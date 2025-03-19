use anyhow::Result;
use clap::Parser;
use log::warn;

use std::path::PathBuf;

use super::set_up_threads;
use deepbiop_fa as fa;
use deepbiop_fq as fq;
use deepbiop_utils as utils;

use deepbiop_core::default;

use fa::encode::Encoder as FaEncoder;
use fq::encode::Encoder as FqEncoder;

#[derive(Debug, Parser)]
pub struct FxToParquet {
    /// path to the fx file
    #[arg(value_name = "fx")]
    fx: PathBuf,

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

fn fa_worker(options: &FxToParquet) -> Result<()> {
    let option = fa::encode::EncoderOptionBuilder::default()
        .bases(fa::encode::BASES.to_vec())
        .build()?;
    let mut fa_encoder = fa::encode::ParquetEncoderBuilder::default()
        .option(option)
        .build()?;

    if options.chunk {
        fa_encoder.encode_chunk(&options.fx, options.chunk_size, false)?;
        return Ok(());
    }

    let (record_batch, schema) = fa_encoder.encode(&options.fx)?;
    // result file is fq_path with .parquet extension
    let parquet_path = if let Some(path) = &options.output {
        if path.with_extension("parquet").exists() {
            warn!("{} already exists, overwriting", path.display());
        }
        path.with_extension("parquet")
    } else {
        options.fx.with_extension("parquet")
    };
    utils::io::write_parquet_for_batches(parquet_path, &record_batch, schema)?;
    Ok(())
}

fn fq_worker(options: &FxToParquet) -> Result<()> {
    let option = fq::encode::EncoderOptionBuilder::default()
        .bases(default::BASES.to_vec())
        .build()?;
    let mut fq_encoder = fq::encode::ParquetEncoderBuilder::default()
        .option(option)
        .build()?;

    if options.chunk {
        fq_encoder.encode_chunk(&options.fx, options.chunk_size, false)?;
        return Ok(());
    }

    let (record_batch, schema) = fq_encoder.encode(&options.fx)?;
    // result file is fq_path with .parquet extension
    let parquet_path = if let Some(path) = &options.output {
        if path.with_extension("parquet").exists() {
            warn!("{} already exists, overwriting", path.display());
        }
        path.with_extension("parquet")
    } else {
        options.fx.with_extension("parquet")
    };
    utils::io::write_parquet_for_batches(parquet_path, &record_batch, schema)?;

    Ok(())
}

impl FxToParquet {
    pub fn run(&self) -> Result<()> {
        use utils::io::SequenceFileType;
        set_up_threads(self.threads)?;
        match utils::io::check_sequence_file_type(&self.fx) {
            Ok(SequenceFileType::Fasta) => fa_worker(self),
            Ok(SequenceFileType::Fastq) => fq_worker(self),
            _ => Err(anyhow::anyhow!("Unsupported file type")),
        }?;

        Ok(())
    }
}
