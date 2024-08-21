use anyhow::Result;
use clap::Parser;
use deepbiop_bam as bam;
use deepbiop_fq as fq;
use noodles::fastq;

use std::path::{Path, PathBuf};

use super::set_up_threads;

#[derive(Debug, Parser)]
pub struct BamToFq {
    /// path to the bam file
    #[arg(value_name = "bam", action=clap::ArgAction::Append)]
    bam: Vec<PathBuf>,

    /// threads number
    #[arg(short, long, default_value = "2")]
    threads: Option<usize>,

    /// output compressed fastq file
    #[arg(short, long, action=clap::ArgAction::SetTrue)]
    compressed: bool,
}

fn write_fq<P: AsRef<Path>>(data: &[fastq::Record], path: P) -> Result<()> {
    let file = std::fs::File::create(path.as_ref())?;
    let mut writer = fastq::io::Writer::new(file);
    for record in data {
        writer.write_record(record)?;
    }
    Ok(())
}

impl BamToFq {
    pub fn run(&self) -> Result<()> {
        set_up_threads(self.threads)?;

        for bam in &self.bam {
            let fq_records = bam::io::bam2fq(bam, self.threads)?;

            if self.compressed {
                let file_path = bam.with_extension("fq.bgz");
                fq::io::write_fq_parallel_for_noodle_record(&fq_records, file_path, self.threads)?;
            } else {
                let file_path = bam.with_extension("fq");
                write_fq(&fq_records, file_path)?;
            }
        }
        Ok(())
    }
}
