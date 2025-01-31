use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;

use super::set_up_threads;

#[derive(Debug, Parser)]
pub struct FaAugment {
    /// path to the bam file
    #[arg(value_name = "fa")]
    fa: PathBuf,

    /// output bgzip compressed file
    #[arg(long, value_name = "output")]
    output: Option<PathBuf>,

    /// threads number
    #[arg(short, long, default_value = "2")]
    threads: Option<usize>,

    /// output bgzip compressed fasta file
    #[arg(short, long, action=clap::ArgAction::SetTrue)]
    compressed: bool,
}

impl FaAugment {
    pub fn run(&self) -> Result<()> {
        set_up_threads(self.threads)?;
        Ok(())
    }
}
