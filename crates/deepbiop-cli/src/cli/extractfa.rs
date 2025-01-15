use ahash::HashSet;
use ahash::HashSetExt;
use anyhow::Result;
use bstr::BString;
use clap::Parser;
use deepbiop_fa as fa;

use log::info;
use std::io::BufRead;
use std::io::BufReader;
use std::path::{Path, PathBuf};

use super::set_up_threads;

#[derive(Debug, Parser)]
pub struct ExtractFa {
    /// path to the bam file
    #[arg(value_name = "fa")]
    fa: PathBuf,

    /// Path to the selected reads
    #[arg(value_name = "reads")]
    reads: PathBuf,

    /// threads number
    #[arg(short, long, default_value = "2")]
    threads: Option<usize>,

    /// output bgzip compressed fasta file
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

impl ExtractFa {
    pub fn run(&self) -> Result<()> {
        set_up_threads(self.threads)?;

        let reads = parse_reads(&self.reads)?;
        info!("load {} selected reads from {:?}", reads.len(), &self.reads);

        let records = fa::io::select_record_from_fa(&self.fa, &reads)?;
        info!("collect {} records", records.len());

        if self.compressed {
            let file_path = self.fa.with_extension("selected.fa.bgz");
            info!("write to {}", &file_path.display());
            fa::io::write_bzip_fa_parallel_for_noodle_record(&records, file_path, self.threads)?;
        } else {
            let file_path = self.fa.with_extension("selected.fa");
            info!("write to {}", &file_path.display());
            fa::io::write_fa_for_noodle_record(&records, file_path)?;
        }

        Ok(())
    }
}
