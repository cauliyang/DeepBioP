use anyhow::Result;
use clap::{Parser, Subcommand};
use needletail::parse_fastx_file;
use std::path::PathBuf;

use deepbiop_utils::export::arrow::SequenceRecord;
use deepbiop_utils::export::numpy::NumpyExporter;
use deepbiop_utils::export::parquet::ParquetWriter;

use super::set_up_threads;

#[derive(Debug, Parser)]
pub struct Export {
    #[command(subcommand)]
    pub command: ExportCommands,
}

#[derive(Debug, Subcommand)]
pub enum ExportCommands {
    /// Export sequences to Parquet format with columnar storage
    Parquet(ParquetCommand),

    /// Export sequences to NumPy format (integer or one-hot encoding)
    Numpy(NumpyCommand),
}

#[derive(Debug, Parser)]
pub struct ParquetCommand {
    /// Input FASTQ/FASTA file
    #[arg(value_name = "INPUT")]
    input: PathBuf,

    /// Output Parquet file
    #[arg(short, long, value_name = "OUTPUT")]
    output: PathBuf,

    /// Number of threads
    #[arg(short = 'j', long, default_value = "2")]
    threads: Option<usize>,
}

impl ParquetCommand {
    pub fn run(&self) -> Result<()> {
        set_up_threads(self.threads)?;

        log::info!("Reading sequences from {:?}", self.input);

        let mut reader = parse_fastx_file(&self.input)?;
        let mut records = Vec::new();

        while let Some(record) = reader.next() {
            let record = record?;
            let id = String::from_utf8_lossy(record.id()).to_string();
            let sequence = record.seq().to_vec();
            let quality = record.qual().map(|q| q.to_vec());

            records.push(SequenceRecord::new(id, sequence, quality));
        }

        log::info!("Loaded {} records", records.len());
        log::info!("Exporting to Parquet: {:?}", self.output);

        // Determine if we have quality scores
        let has_quality = records.iter().any(|r| r.quality.is_some());

        let writer = if has_quality {
            ParquetWriter::for_fastq()
        } else {
            ParquetWriter::for_fasta()
        };

        writer.write(&self.output, &records)?;

        log::info!("Successfully exported {} records to Parquet", records.len());
        Ok(())
    }
}

#[derive(Debug, Parser)]
pub struct NumpyCommand {
    /// Input FASTQ/FASTA file
    #[arg(value_name = "INPUT")]
    input: PathBuf,

    /// Output .npy file
    #[arg(short, long, value_name = "OUTPUT")]
    output: PathBuf,

    /// Encoding type: 'integer' (A=0, C=1, G=2, T=3) or 'onehot' (binary matrix)
    #[arg(short = 'e', long, default_value = "integer")]
    encoding: String,

    /// Alphabet for encoding (default: ACGT for DNA)
    #[arg(short = 'a', long, default_value = "ACGT")]
    alphabet: String,

    /// Number of threads
    #[arg(short = 'j', long, default_value = "2")]
    threads: Option<usize>,
}

impl NumpyCommand {
    pub fn run(&self) -> Result<()> {
        set_up_threads(self.threads)?;

        log::info!("Reading sequences from {:?}", self.input);

        let mut reader = parse_fastx_file(&self.input)?;
        let mut sequences = Vec::new();

        while let Some(record) = reader.next() {
            let record = record?;
            sequences.push(record.seq().to_vec());
        }

        log::info!("Loaded {} sequences", sequences.len());
        log::info!("Exporting to NumPy with {} encoding", self.encoding);

        let exporter = NumpyExporter;
        let alphabet = self.alphabet.as_bytes();

        match self.encoding.as_str() {
            "integer" => {
                exporter.export_integer(&self.output, &sequences, alphabet)?;
                log::info!(
                    "Successfully exported {} sequences as integer encoding",
                    sequences.len()
                );
            }
            "onehot" => {
                exporter.export_onehot(&self.output, &sequences, alphabet)?;
                log::info!(
                    "Successfully exported {} sequences as one-hot encoding",
                    sequences.len()
                );
            }
            _ => {
                return Err(anyhow::anyhow!(
                    "Invalid encoding type: {}. Use 'integer' or 'onehot'",
                    self.encoding
                ));
            }
        }

        Ok(())
    }
}

impl Export {
    pub fn run(&self) -> Result<()> {
        match &self.command {
            ExportCommands::Parquet(cmd) => cmd.run(),
            ExportCommands::Numpy(cmd) => cmd.run(),
        }
    }
}
