use anyhow::Result;
use clap::Parser;
use deepbiop_fq as fq;

use std::path::PathBuf;

use super::set_up_threads;

#[derive(Debug, Parser)]
pub struct FqsToOne {
    /// path to the fq file
    #[arg(value_name = "fqs", action=clap::ArgAction::Append)]
    fqs: Vec<PathBuf>,

    /// output bgzip compressed file
    #[arg(long, value_name = "output")]
    output: PathBuf,

    #[arg(short, long, default_value = "2")]
    threads: Option<usize>,
}

impl FqsToOne {
    pub fn run(&self) -> Result<()> {
        set_up_threads(self.threads)?;
        let output = self.output.with_extension("fq.gz");
        fq::io::convert_multiple_fqs_to_one_bgzip_fq(&self.fqs, output, true)?;
        Ok(())
    }
}
