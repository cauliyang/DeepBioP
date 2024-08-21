use anyhow::Result;
use clap::Parser;
use deepbiop_bam as bam;
use deepbiop_fq as fq;

use std::path::PathBuf;

use super::set_up_threads;

#[derive(Debug, Parser)]
pub struct BamToFq {
    /// path to the bam file
    #[arg(value_name = "bam", action=clap::ArgAction::Append)]
    bam: Vec<PathBuf>,

    /// threads number
    #[arg(short, long, default_value = "2")]
    threads: Option<usize>,
}

impl BamToFq {
    pub fn run(&self) -> Result<()> {
        set_up_threads(self.threads)?;

        for bam in &self.bam {
            let fq_records = bam::io::bam2fq(bam, self.threads)?;
            let file_path = bam.with_extension("fq.bgz");
            fq::io::write_fq_parallel_for_noodle_record(&fq_records, file_path, self.threads)?;
        }
        Ok(())
    }
}
