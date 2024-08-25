use anyhow::Result;
use clap::Parser;

use noodles::fasta;
use noodles::fastq;
use std::path::PathBuf;

use super::set_up_threads;

#[derive(Debug, Parser)]
pub struct FaToFq {
    /// path to the fa file
    #[arg(value_name = "fa", action=clap::ArgAction::Append)]
    fa: Vec<PathBuf>,

    /// threads number
    #[arg(short, long, default_value = "2")]
    threads: Option<usize>,
}

impl FaToFq {
    pub fn run(&self) -> Result<()> {
        set_up_threads(self.threads)?;

        for fa in &self.fa {
            let fq_file_path = fa.with_extension("fa");

            let mut reader = fasta::io::reader::Builder.build_from_path(fa)?;

            let fq_writer_handle = std::fs::File::create(fq_file_path)?;
            let mut fq_writer = fastq::io::Writer::new(fq_writer_handle);

            for record in reader.records() {
                let record = record?;
                let name = record.name();
                let sequence = record.sequence().as_ref().to_vec();
                let quality = vec![b'@'; sequence.len()];
                let fq_record = fastq::Record::new(
                    fastq::record::Definition::new(name.to_vec(), ""),
                    sequence,
                    quality,
                );
                fq_writer.write_record(&fq_record)?;
            }
        }
        Ok(())
    }
}
