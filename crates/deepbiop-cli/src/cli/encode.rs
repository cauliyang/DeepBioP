use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

use deepbiop_core::kmer::encode::KmerEncoder;
use deepbiop_core::types::EncodingType;
use deepbiop_fq::encode::integer::IntegerEncoder;
use deepbiop_fq::encode::onehot::{AmbiguousStrategy, OneHotEncoder};
use needletail::parse_fastx_file;

use super::set_up_threads;

#[derive(Debug, Parser)]
pub struct Encode {
    #[command(subcommand)]
    pub command: EncodeCommands,
}

#[derive(Debug, Subcommand)]
pub enum EncodeCommands {
    /// One-hot encode sequences from FASTQ/FASTA files
    Onehot(OnehotCommand),

    /// K-mer encode sequences from FASTQ/FASTA files
    Kmer(KmerCommand),

    /// Integer encode sequences from FASTQ/FASTA files
    Integer(IntegerCommand),
}

#[derive(Debug, Clone, ValueEnum)]
pub enum EncodingTypeArg {
    Dna,
    Rna,
    Protein,
}

impl From<EncodingTypeArg> for EncodingType {
    fn from(arg: EncodingTypeArg) -> Self {
        match arg {
            EncodingTypeArg::Dna => EncodingType::DNA,
            EncodingTypeArg::Rna => EncodingType::RNA,
            EncodingTypeArg::Protein => EncodingType::Protein,
        }
    }
}

#[derive(Debug, Clone, ValueEnum)]
pub enum AmbiguousStrategyArg {
    Skip,
    Mask,
    Random,
}

impl From<AmbiguousStrategyArg> for AmbiguousStrategy {
    fn from(arg: AmbiguousStrategyArg) -> Self {
        match arg {
            AmbiguousStrategyArg::Skip => AmbiguousStrategy::Skip,
            AmbiguousStrategyArg::Mask => AmbiguousStrategy::Mask,
            AmbiguousStrategyArg::Random => AmbiguousStrategy::Random,
        }
    }
}

#[derive(Debug, Parser)]
pub struct OnehotCommand {
    /// Input FASTQ/FASTA file
    #[arg(value_name = "INPUT")]
    input: PathBuf,

    /// Output file (NumPy .npy format)
    #[arg(short, long, value_name = "OUTPUT")]
    output: PathBuf,

    /// Encoding type
    #[arg(short = 't', long, value_enum, default_value = "dna")]
    encoding_type: EncodingTypeArg,

    /// Strategy for handling ambiguous bases
    #[arg(short = 's', long, value_enum, default_value = "skip")]
    ambiguous_strategy: AmbiguousStrategyArg,

    /// Random seed for reproducible random ambiguity handling
    #[arg(long)]
    seed: Option<u64>,

    /// Number of threads
    #[arg(short = 'j', long, default_value = "2")]
    threads: Option<usize>,
}

impl OnehotCommand {
    pub fn run(&self) -> Result<()> {
        set_up_threads(self.threads)?;

        let encoding_type: EncodingType = self.encoding_type.clone().into();
        let ambiguous_strategy: AmbiguousStrategy = self.ambiguous_strategy.clone().into();

        let encoder = if let Some(seed) = self.seed {
            OneHotEncoder::with_seed(encoding_type, ambiguous_strategy, seed)
        } else {
            OneHotEncoder::new(encoding_type, ambiguous_strategy)
        };

        // Read sequences from input file
        let mut reader = parse_fastx_file(&self.input)?;
        let mut sequences = Vec::new();

        while let Some(record) = reader.next() {
            let record = record?;
            sequences.push(record.seq().to_vec());
        }

        // Encode sequences
        let seq_refs: Vec<&[u8]> = sequences.iter().map(|s| s.as_slice()).collect();
        let encoded = encoder.encode_batch(&seq_refs)?;

        // Save to NumPy file
        log::info!(
            "Encoded {} sequences with shape {:?}",
            sequences.len(),
            encoded.shape()
        );
        log::info!("Saving to {:?}", self.output);

        // TODO: Implement NumPy .npy file writing
        // For now, just print the shape
        println!("Encoded shape: {:?}", encoded.shape());
        println!("Output would be saved to: {:?}", self.output);

        Ok(())
    }
}

#[derive(Debug, Parser)]
pub struct KmerCommand {
    /// Input FASTQ/FASTA file
    #[arg(value_name = "INPUT")]
    input: PathBuf,

    /// Output file (NumPy .npy format)
    #[arg(short, long, value_name = "OUTPUT")]
    output: PathBuf,

    /// K-mer size
    #[arg(short, long, default_value = "3")]
    k: usize,

    /// Use canonical k-mers (treat forward and reverse complement as same)
    #[arg(short, long, default_value = "true")]
    canonical: bool,

    /// Encoding type
    #[arg(short = 't', long, value_enum, default_value = "dna")]
    encoding_type: EncodingTypeArg,

    /// Number of threads
    #[arg(short = 'j', long, default_value = "2")]
    threads: Option<usize>,
}

impl KmerCommand {
    pub fn run(&self) -> Result<()> {
        set_up_threads(self.threads)?;

        let encoding_type: EncodingType = self.encoding_type.clone().into();
        let encoder = KmerEncoder::new(self.k, self.canonical, encoding_type);

        // Read sequences from input file
        let mut reader = parse_fastx_file(&self.input)?;
        let mut sequences = Vec::new();

        while let Some(record) = reader.next() {
            let record = record?;
            sequences.push(record.seq().to_vec());
        }

        // Encode sequences
        let seq_refs: Vec<&[u8]> = sequences.iter().map(|s| s.as_slice()).collect();
        let encoded = encoder.encode_batch(&seq_refs)?;

        // Save to NumPy file
        log::info!(
            "Encoded {} sequences with shape {:?}",
            sequences.len(),
            encoded.shape()
        );
        log::info!("Saving to {:?}", self.output);

        // TODO: Implement NumPy .npy file writing
        // For now, just print the shape
        println!("Encoded shape: {:?}", encoded.shape());
        println!("Output would be saved to: {:?}", self.output);

        Ok(())
    }
}

#[derive(Debug, Parser)]
pub struct IntegerCommand {
    /// Input FASTQ/FASTA file
    #[arg(value_name = "INPUT")]
    input: PathBuf,

    /// Output file (NumPy .npy format)
    #[arg(short, long, value_name = "OUTPUT")]
    output: PathBuf,

    /// Encoding type
    #[arg(short = 't', long, value_enum, default_value = "dna")]
    encoding_type: EncodingTypeArg,

    /// Number of threads
    #[arg(short = 'j', long, default_value = "2")]
    threads: Option<usize>,
}

impl IntegerCommand {
    pub fn run(&self) -> Result<()> {
        set_up_threads(self.threads)?;

        let encoding_type: EncodingType = self.encoding_type.clone().into();
        let encoder = IntegerEncoder::new(encoding_type);

        // Read sequences from input file
        let mut reader = parse_fastx_file(&self.input)?;
        let mut sequences = Vec::new();

        while let Some(record) = reader.next() {
            let record = record?;
            sequences.push(record.seq().to_vec());
        }

        // Encode sequences
        let seq_refs: Vec<&[u8]> = sequences.iter().map(|s| s.as_slice()).collect();
        let encoded = encoder.encode_batch(&seq_refs)?;

        // Save to NumPy file
        log::info!(
            "Encoded {} sequences with shape {:?}",
            sequences.len(),
            encoded.shape()
        );
        log::info!("Saving to {:?}", self.output);

        // TODO: Implement NumPy .npy file writing
        // For now, just print the shape
        println!("Encoded shape: {:?}", encoded.shape());
        println!("Output would be saved to: {:?}", self.output);

        Ok(())
    }
}

impl Encode {
    pub fn run(&self) -> Result<()> {
        match &self.command {
            EncodeCommands::Onehot(cmd) => cmd.run(),
            EncodeCommands::Kmer(cmd) => cmd.run(),
            EncodeCommands::Integer(cmd) => cmd.run(),
        }
    }
}
