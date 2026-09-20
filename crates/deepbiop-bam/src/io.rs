use anyhow::{Context, Result};
use noodles::{bam, bgzf};
use rayon::prelude::*;
use std::{fs::File, path::Path};

use deepbiop_utils as utils;
use noodles::fastq;

// FIXME: The function has a bug since seq != qual

pub fn bam2fq(bam: &Path, threads: Option<usize>) -> Result<Vec<fastq::Record>> {
    let worker_count = utils::parallel::calculate_worker_count(threads);
    let file = File::open(bam)?;
    let decoder = bgzf::io::MultithreadedReader::with_worker_count(worker_count, file);
    let mut reader = bam::io::Reader::from(decoder);
    let _header = reader.read_header()?;

    let records = reader
        .records()
        .collect::<std::io::Result<Vec<_>>>()
        .context("Failed to read BAM records")?;

    // Conversion is parallel but index-preserving, so the FASTQ output keeps the
    // BAM record order.
    records
        .into_par_iter()
        .map(|record| {
            let seq = record.sequence().as_ref().to_vec();
            let qual = record.quality_scores().as_ref().to_vec();

            if seq.len() != qual.len() {
                let name =
                    String::from_utf8_lossy(record.name().unwrap_or_default().as_ref()).to_string();
                return Err(anyhow::anyhow!(
                    "{} seq and qual length are not equal",
                    name
                ));
            }

            let fq_record = fastq::Record::new(
                fastq::record::Definition::new(record.name().unwrap_or_default().to_vec(), ""),
                seq,
                qual,
            );
            Ok(fq_record)
        })
        .collect::<Result<Vec<fastq::Record>>>()
}
