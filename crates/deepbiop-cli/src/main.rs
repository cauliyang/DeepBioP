use clap::{Command, CommandFactory, Parser, Subcommand};
use env_logger::Builder;
use human_panic::setup_panic;
use log::info;
use log::LevelFilter;
use std::fmt::Display;
use std::io::Result;

use clap_complete::{generate, Generator, Shell};
use std::io;

mod cli;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
#[command(propagate_version = true)]
struct Cli {
    // If provided, outputs the completion file for given shell
    #[arg(long = "generate", value_enum)]
    generator: Option<Shell>,

    #[command(subcommand)]
    command: Option<Commands>,

    #[command(flatten)]
    verbose: clap_verbosity_flag::Verbosity,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Count chimeric reads in a BAM file.
    CountChimeric(cli::CountChimeric),

    /// BAM to fastq conversion.
    BamToFq(cli::BamToFq),

    /// Fastq to fasta conversion.
    FqToFa(cli::FqToFa),

    /// Fasta to fastq conversion.
    FaToFq(cli::FaToFq),

    /// Fasta to parquet conversion.
    FaToParquet(cli::FaToParquet),
}

impl Display for Commands {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Commands::CountChimeric(_) => write!(f, "chimericcount"),
            Commands::BamToFq(_) => write!(f, "bam2fq"),
            Commands::FqToFa(_) => write!(f, "fq2fa"),
            Commands::FaToFq(_) => write!(f, "fa2fq"),
            Commands::FaToParquet(_) => write!(f, "fa2parquet"),
        }
    }
}

fn print_completions<G: Generator>(gen: G, cmd: &mut Command) {
    generate(gen, cmd, cmd.get_name().to_string(), &mut io::stdout());
}

fn main() -> Result<()> {
    setup_panic!();

    let start = std::time::Instant::now();
    let cli = Cli::parse();

    let mut log_builder = Builder::from_default_env();

    match cli.verbose.log_level() {
        Some(level) => {
            info!("Verbose mode is on with level {}!", level);
            log_builder.filter(None, level.to_level_filter());
        }
        None => {
            info!("Verbose mode is off!");
            log_builder.filter(None, LevelFilter::Off);
        }
    }
    log_builder.init();

    if let Some(generator) = cli.generator {
        let mut cmd = Cli::command();
        info!("Generating completion file for {generator:?}...");
        print_completions(generator, &mut cmd);
        return Ok(());
    }

    match &cli.command {
        Some(Commands::CountChimeric(count_chimeric)) => {
            count_chimeric.run().unwrap();
        }

        Some(Commands::BamToFq(bam2fq)) => {
            bam2fq.run().unwrap();
        }

        Some(Commands::FqToFa(fq2fa)) => {
            fq2fa.run().unwrap();
        }

        Some(Commands::FaToFq(fa2fq)) => {
            fa2fq.run().unwrap();
        }

        Some(Commands::FaToParquet(fa2parquet)) => {
            fa2parquet.run().unwrap();
        }

        None => {
            println!("No command provided!");
        }
    }

    let elapsed = start.elapsed();
    log::info!("elapsed time: {:.2?}", elapsed);
    Ok(())
}
