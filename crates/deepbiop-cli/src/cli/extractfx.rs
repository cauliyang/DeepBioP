use ahash::HashSet;
use ahash::HashSetExt;
use anyhow::Result;
use bstr::BString;
use clap::Parser;
use deepbiop_fa as fa;
use deepbiop_fq as fq;
use deepbiop_utils as utils;

use log::info;
use std::io::BufRead;
use std::io::BufReader;
use std::path::{Path, PathBuf};

use super::set_up_threads;

#[derive(Debug, Parser)]
pub struct ExtractFx {
    /// path to the fastx file
    #[arg(value_name = "fx")]
    fx: PathBuf,

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

fn fa_worker(options: &ExtractFx) -> Result<()> {
    let records = if let Some(reads_path) = &options.reads {
        let reads = parse_reads(reads_path)?;
        let records = fa::io::select_record_from_fa(&options.fx, &reads)?;
        info!("load {} selected reads from {:?}", reads.len(), reads_path);
        records
    } else if let Some(number) = options.number {
        let records = fa::io::select_record_from_fq_by_random(&options.fx, number)?;
        info!("select {} reads by random", number);
        records
    } else {
        return Err(anyhow::anyhow!(
            "Either --reads or --number must be specified"
        ));
    };

    info!("collect {} records", records.len());

    if options.compressed {
        let file_path = if let Some(path) = &options.output {
            let path = path.with_extension("fa.gz");
            if path.exists() {
                info!("{} already exists, overwriting", path.display());
            }
            path
        } else {
            options.fx.with_extension("selected.fa.gz")
        };

        info!("write to {}", &file_path.display());
        fa::io::write_bzip_fa_parallel_for_noodle_record(&records, file_path, options.threads)?;
    } else {
        let file_path = if let Some(path) = &options.output {
            let path = path.with_extension("fa");
            if path.exists() {
                info!("{} already exists, overwriting", path.display());
            }
            path
        } else {
            options.fx.with_extension("selected.fa")
        };
        info!("write to {}", &file_path.display());
        fa::io::write_fa_for_noodle_record(&records, file_path)?;
    }
    Ok(())
}

fn fq_worker(options: &ExtractFx) -> Result<()> {
    let records = if let Some(reads_path) = &options.reads {
        let reads = parse_reads(reads_path)?;
        let records = fq::io::select_record_from_fq(&options.fx, &reads)?;

        info!("load {} selected reads from {:?}", reads.len(), reads_path);
        records
    } else if let Some(number) = options.number {
        let records = fq::io::select_record_from_fq_by_random(&options.fx, number)?;
        info!("select {} reads by random", number);
        records
    } else {
        return Err(anyhow::anyhow!(
            "Either --reads or --number must be specified"
        ));
    };

    info!("collect {} records", records.len());

    if options.compressed {
        let file_path = if let Some(path) = &options.output {
            let path = path.with_extension("fq.gz");
            if path.exists() {
                info!("{} already exists, overwriting", path.display());
            }
            path
        } else {
            options.fx.with_extension("selected.fq.gz")
        };
        info!("write to {}", &file_path.display());
        fq::io::write_bgzip_fq_parallel_for_noodle_record(&records, file_path, options.threads)?;
    } else {
        let file_path = if let Some(path) = &options.output {
            let path = path.with_extension("fq");
            if path.exists() {
                info!("{} already exists, overwriting", path.display());
            }
            path
        } else {
            options.fx.with_extension("selected.fq")
        };
        info!("write to {}", &file_path.display());
        fq::io::write_fq_for_noodle_record(&records, file_path)?;
    }
    Ok(())
}

impl ExtractFx {
    pub fn run(&self) -> Result<()> {
        set_up_threads(self.threads)?;
        match utils::io::check_sequence_file_type(&self.fx) {
            Ok(utils::io::SequenceFileType::Fasta) => fa_worker(self),
            Ok(utils::io::SequenceFileType::Fastq) => fq_worker(self),
            _ => Err(anyhow::anyhow!("Unsupported file type")),
        }?;
        Ok(())
    }
}
