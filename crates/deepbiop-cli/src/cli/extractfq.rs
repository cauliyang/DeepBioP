use ahash::HashSet;
use ahash::HashSetExt;
use anyhow::Result;
use bstr::BString;
use clap::Parser;
use deepbiop_fq as fq;

use super::set_up_threads;
use log::info;
use std::io::BufRead;
use std::io::BufReader;
use std::path::{Path, PathBuf};

#[derive(Debug, Parser)]
pub struct ExtractFq {
    /// path to the fq file
    #[arg(value_name = "fq")]
    fq: PathBuf,

    /// Path to the selected reads
    #[arg(long, value_name = "reads", conflicts_with = "number")]
    reads: Option<PathBuf>,

    /// The number of selected reads by random
    #[arg(long, value_name = "number", conflicts_with = "reads")]
    number: Option<usize>,

    /// output bgzip compressed file
    #[arg(long, value_name = "output")]
    output: Option<PathBuf>,

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

        let records = if let Some(reads_path) = &self.reads {
            let reads = parse_reads(reads_path)?;
            let records = fq::io::select_record_from_fq(&self.fq, &reads)?;

            info!("load {} selected reads from {:?}", reads.len(), reads_path);
            records
        } else if let Some(number) = self.number {
            let records = fq::io::select_record_from_fq_by_random(&self.fq, number)?;
            info!("select {} reads by random", number);
            records
        } else {
            return Err(anyhow::anyhow!(
                "Either --reads or --number must be specified"
            ));
        };

        info!("collect {} records", records.len());

        if self.compressed {
            let file_path = if let Some(path) = &self.output {
                let path = path.with_extension("fq.gz");
                if path.exists() {
                    info!("{} already exists, overwriting", path.display());
                }
                path
            } else {
                self.fq.with_extension("selected.fq.gz")
            };
            info!("write to {}", &file_path.display());
            fq::io::write_bgzip_fq_parallel_for_noodle_record(&records, file_path, self.threads)?;
        } else {
            let file_path = if let Some(path) = &self.output {
                let path = path.with_extension("fq");
                if path.exists() {
                    info!("{} already exists, overwriting", path.display());
                }
                path
            } else {
                self.fq.with_extension("selected.fq")
            };
            info!("write to {}", &file_path.display());
            fq::io::write_fq_for_noodle_record(&records, file_path)?;
        }
        Ok(())
    }
}
