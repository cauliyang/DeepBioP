use ahash::HashMap;
use anyhow::Result;
use noodles::bam;
use noodles::bgzf;
use rayon::prelude::*;
use std::path::{Path, PathBuf};
use std::{fs::File, num::NonZeroUsize, thread};

use noodles::sam::alignment::record::data::field::Tag;
use noodles::sam::alignment::record::data::field::Value;

pub fn count_chimeric_reads_for_paths(
    bams: &[PathBuf],
    threads: Option<usize>,
) -> HashMap<PathBuf, usize> {
    bams.par_iter()
        .filter_map(|path| match count_chimeric_reads_for_path(path, threads) {
            Ok(count) => Some((path.clone(), count)),
            Err(e) => {
                eprintln!(
                    "Error counting chimeric reads for {}: {}",
                    path.display(),
                    e
                );
                None
            }
        })
        .collect()
}

pub fn count_chimeric_reads_for_path<P: AsRef<Path>>(
    bam: P,
    threads: Option<usize>,
) -> Result<usize> {
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

    let res = reader
        .records()
        .par_bridge()
        .filter_map(|result| {
            let record = result.unwrap();
            let is_mapped = !record.flags().is_unmapped();
            let is_not_secondary = !record.flags().is_secondary();
            let is_primary = !record.flags().is_supplementary();

            let has_sa_tag = matches!(
                record.data().get(&Tag::OTHER_ALIGNMENTS),
                Some(Ok(Value::String(_sa_string)))
            );

            if is_mapped && is_not_secondary && is_primary && has_sa_tag {
                Some(record)
            } else {
                None
            }
        })
        .count();
    Ok(res)
}
