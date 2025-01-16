use ahash::HashSet;
use ahash::HashSetExt;
use anyhow::Result;
use bstr::BString;
use clap::Parser;
use deepbiop_fq as fq;

use std::io::BufRead;
use std::io::BufReader;
use std::path::{Path, PathBuf};

use super::set_up_threads;
use log::info;

#[derive(Debug, Parser)]
pub struct ExtractFq {
    /// path to the bam file
    #[arg(value_name = "fq")]
    fq: PathBuf,

    /// Path to the selected reads
    #[arg(value_name = "reads")]
    reads: PathBuf,

    /// threads number
    #[arg(short, long, default_value = "2")]
    threads: Option<usize>,

    /// output bgzip compressed fastq file
    #[arg(short, long, action=clap::ArgAction::SetTrue)]
    compressed: bool,
}

fn parse_reads<P: AsRef<Path>>(reads: P) -> Result<HashSet<BString>> {
    let file = std::fs::File::open(reads.as_ref())?;

    let reader = BufReader::new(file);

    let mut reads = HashSet::new();

    for line in reader.lines() {
        let line = line?;
        reads.insert(line.into());
    }

    Ok(reads)
}

impl ExtractFq {
    pub fn run(&self) -> Result<()> {
        set_up_threads(self.threads)?;

        let reads = parse_reads(&self.reads)?;
        info!("load {} selected reads from {:?}", reads.len(), &self.reads);

        let records = fq::io::select_record_from_fq(&self.fq, &reads)?;
        info!("collect {} records", records.len());

        if self.compressed {
            let file_path = self.fq.with_extension("selected.fq.bgz");
            info!("write to {}", &file_path.display());
            fq::io::write_bgzip_fq_parallel_for_noodle_record(&records, file_path, self.threads)?;
        } else {
            let file_path = self.fq.with_extension("selected.fq");
            info!("write to {}", &file_path.display());
            fq::io::write_fq_for_noodle_record(&records, file_path)?;
        }
        Ok(())
    }
}
