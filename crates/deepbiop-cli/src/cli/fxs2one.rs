use anyhow::Result;
use clap::Parser;
use deepbiop_fa as fa;
use deepbiop_fq as fq;
use deepbiop_utils as utils;

use std::path::PathBuf;

use super::set_up_threads;

#[derive(Debug, Parser)]
pub struct FxsToOne {
    /// path to the fx file
    #[arg(value_name = "fxs", action=clap::ArgAction::Append)]
    fxs: Vec<PathBuf>,

    /// output bgzip compressed file
    #[arg(long, value_name = "output")]
    output: PathBuf,

    #[arg(short, long, default_value = "2")]
    threads: Option<usize>,
}

impl FxsToOne {
    pub fn run(&self) -> Result<()> {
        set_up_threads(self.threads)?;
        match utils::io::check_sequence_file_type(&self.fxs[0]) {
            Ok(utils::io::SequenceFileType::Fasta) => {
                let output = self.output.with_extension("fa.gz");
                fa::io::convert_multiple_fas_to_one_bgzip_fa(&self.fxs, output, true)?;
            }
            Ok(utils::io::SequenceFileType::Fastq) => {
                let output = self.output.with_extension("fq.gz");
                fq::io::convert_multiple_fqs_to_one_bgzip_fq(&self.fxs, output, true)?;
            }

            _ => {
                return Err(anyhow::anyhow!("Unsupported file type"));
            }
        }
        Ok(())
    }
}
