use anyhow::Result;
use clap::Parser;

use std::path::PathBuf;

use super::set_up_threads;

#[derive(Debug, Parser)]
pub struct FaToParquet {
    /// path to the fa file
    #[arg(value_name = "fa")]
    fa: PathBuf,

    /// threads number
    #[arg(short, long, default_value = "2")]
    threads: Option<usize>,
}

impl FaToParquet {
    pub fn run(&self) -> Result<()> {
        set_up_threads(self.threads)?;

        Ok(())
    }
}
