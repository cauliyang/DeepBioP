use anyhow::Result;
use clap::Parser;
use deepbiop_bam as bam;
use std::path::PathBuf;

use super::set_up_threads;

#[derive(Debug, Parser)]
pub struct CountChimeric {
    /// path to the bam file
    #[arg(value_name = "bam", action=clap::ArgAction::Append)]
    bam: Vec<PathBuf>,

    /// threads number
    #[arg(short, long, default_value = "2")]
    threads: Option<usize>,
}

impl CountChimeric {
    pub fn run(&self) -> Result<()> {
        set_up_threads(self.threads)?;
        let res = bam::chimeric::count_chimeric_reads_for_paths(&self.bam, self.threads);
        for (path, count) in res {
            log::info!("{}: {}", path.to_string_lossy(), count);
        }

        Ok(())
    }
}
