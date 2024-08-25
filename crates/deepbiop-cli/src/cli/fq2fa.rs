use anyhow::Result;
use clap::Parser;
use deepbiop_fq as fq;

use noodles::fasta;
use std::path::PathBuf;

use super::set_up_threads;

#[derive(Debug, Parser)]
pub struct FqToFa {
    /// path to the fq file
    #[arg(value_name = "fq", action=clap::ArgAction::Append)]
    fq: Vec<PathBuf>,

    /// threads number
    #[arg(short, long, default_value = "2")]
    threads: Option<usize>,
}

impl FqToFa {
    pub fn run(&self) -> Result<()> {
        set_up_threads(self.threads)?;

        for fq in &self.fq {
            let fq_records = fq::io::fastq_to_fasta(fq)?;
            let fa_file_path = fq.with_extension("fa");
            let fa_file_handler = std::fs::File::create(&fa_file_path)?;
            let mut fa_writer = fasta::io::Writer::new(fa_file_handler);
            for record in fq_records {
                fa_writer.write_record(&record)?;
            }
        }
        Ok(())
    }
}
