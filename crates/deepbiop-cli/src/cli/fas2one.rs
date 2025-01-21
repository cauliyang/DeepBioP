use anyhow::Result;
use clap::Parser;
use deepbiop_fa as fa;

use std::path::PathBuf;

use super::set_up_threads;

#[derive(Debug, Parser)]
pub struct FasToOne {
    /// path to the fa file
    #[arg(value_name = "fas", action=clap::ArgAction::Append)]
    fas: Vec<PathBuf>,

    /// output bgzip compressed file
    #[arg(long, value_name = "output")]
    output: PathBuf,

    #[arg(short, long, default_value = "2")]
    threads: Option<usize>,
}

impl FasToOne {
    pub fn run(&self) -> Result<()> {
        set_up_threads(self.threads)?;
        fa::io::convert_multiple_fas_to_one_bgzip_fa(&self.fas, &self.output, true)?;
        Ok(())
    }
}
