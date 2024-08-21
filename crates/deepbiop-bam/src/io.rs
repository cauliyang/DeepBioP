use anyhow::Result;
use noodles::{bam, bgzf};
use rayon::prelude::*;
use std::{fs::File, num::NonZeroUsize, path::Path, thread};

use noodles::fastq;

pub fn bam2fq(bam: &Path, threads: Option<usize>) -> Result<Vec<fastq::Record>> {
    let file = File::open(bam)?;
    let worker_count = if let Some(threads) = threads {
        NonZeroUsize::new(threads)
            .unwrap()
            .min(thread::available_parallelism().unwrap_or(NonZeroUsize::MIN))
    } else {
        thread::available_parallelism().unwrap_or(NonZeroUsize::MIN)
    };

    let decoder = bgzf::MultithreadedReader::with_worker_count(worker_count, file);
    let mut reader = bam::io::Reader::from(decoder);
    let _header = reader.read_header()?;

    Ok(reader
        .records()
        .par_bridge()
        .map(|result| {
            let record = result.unwrap();
            let fq_record = fastq::Record::new(
                fastq::record::Definition::new(record.name().unwrap().to_vec(), ""),
                record.sequence().as_ref().to_vec(),
                record.quality_scores().as_ref().to_vec(),
            );
            fq_record
        })
        .collect::<Vec<fastq::Record>>())
}
