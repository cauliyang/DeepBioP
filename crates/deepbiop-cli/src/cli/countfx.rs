use std::{
    fs::File,
    io::{BufReader, BufWriter, Write},
    path::{Path, PathBuf},
};

use clap::Parser;
use deepbiop_utils as utils;
use noodles::{fasta, fastq};

use super::set_up_threads;
use anyhow::Result;
use rayon::prelude::*;

#[derive(Debug, Parser)]
pub struct CountFx {
    /// path to the fastx file
    #[arg(value_name = "fx")]
    fx: PathBuf,

    /// if export the result
    #[arg(long, action=clap::ArgAction::SetTrue)]
    export: bool,

    /// threads number
    #[arg(short, long, default_value = "2")]
    threads: Option<usize>,

    // query length
    #[arg(long,  default_value = "0")]
    query_length: usize,
}

fn export_json<P: AsRef<Path>>(seq_len: &[usize], output: P) -> Result<()> {
    // save identities to json file
    let json_file = File::create(output)?;
    let mut json_writer = BufWriter::new(json_file);
    json_writer.write_all(serde_json::to_string(seq_len)?.as_bytes())?;
    Ok(())
}

fn summary<P: AsRef<Path>>(seq_len: &[usize], output: P, export: bool, query_length: usize) -> Result<()> {
    let total_len: usize = seq_len.par_iter().sum();
    let max_len: usize = *seq_len.par_iter().max().unwrap_or(&0);
    let min_len: usize = *seq_len.par_iter().min().unwrap_or(&0);
    let mean_len: f64 = total_len as f64 / seq_len.len() as f64;

    // Sort the lengths to calculate quartiles
    let mut sorted_lens = seq_len.to_vec();
    sorted_lens.par_sort_unstable();

    let n = sorted_lens.len();
    let q1_idx = n / 4;
    let q2_idx = n / 2;
    let q3_idx = 3 * n / 4;
    let p90_idx = n * 90 / 100;
    let p95_idx = n * 95 / 100;
    let p99_idx = n * 99 / 100;

    let q1 = sorted_lens[q1_idx];
    let q2 = sorted_lens[q2_idx]; // median
    let q3 = sorted_lens[q3_idx];

    // get 90th percentile and 95th percentile and 99th percentile
    let p90 = sorted_lens[p90_idx];
    let p95 = sorted_lens[p95_idx];
    let p99 = sorted_lens[p99_idx];

    // Calculate std dev in parallel
    let variance = seq_len
        .par_iter()
        .map(|&len| {
            let diff = len as f64 - mean_len;
            diff * diff
        })
        .sum::<f64>()
        / seq_len.len() as f64;
    let std_dev = variance.sqrt();

    println!("The number of sequences           : {}", seq_len.len());
    println!("The minimum length of sequences   : {}", min_len);
    println!("The maximum length of sequences   : {}", max_len);
    println!("The mean length of sequences      : {:.2}", mean_len);
    println!("The standard deviation of sequences: {:.2}", std_dev);

    println!("The first quartile of sequences   : {}", q1);
    println!("The second quartile of sequences  : {}", q2);
    println!("The third quartile of sequences   : {}", q3);

    println!("The 90th percentile of sequences  : {}", p90);
    println!("The 95th percentile of sequences  : {}", p95);
    println!("The 99th percentile of sequences  : {}", p99);

    if query_length > 0 {
        match sorted_lens.binary_search(&query_length) {
            Ok(idx) | Err(idx) => {
                let count = sorted_lens.len() - idx;
                println!("The number of sequences with length >= {query_length}: {count}");
                println!("The percentage of sequences with length >= {query_length}: {:.2}%", count as f64 / seq_len.len() as f64 * 100.0);
            }
        }

    }

    if export {
        export_json(&sorted_lens, output)?;
    }
    Ok(())
}

fn count_fx<P: AsRef<Path>>(fx: P, export: bool, query_length: usize) -> Result<()> {
    use utils::io::SequenceFileType;
    let reader = utils::io::create_reader_for_compressed_file(&fx)?;

    let seq_len = match utils::io::check_sequence_file_type(&fx)? {
        SequenceFileType::Fasta => {
            let mut reader = fasta::io::Reader::new(BufReader::new(reader));
            reader
                .records()
                .map(|record| record.map(|r| r.sequence().len()))
                .collect::<Result<Vec<_>, _>>()?
        }
        SequenceFileType::Fastq => {
            let mut reader = fastq::io::Reader::new(BufReader::new(reader));
            reader
                .records()
                .map(|record| record.map(|r| r.sequence().len()))
                .collect::<Result<Vec<_>, _>>()?
        }
        _ => return Err(anyhow::anyhow!("Unsupported file type")),
    };

    let output = fx.as_ref().with_extension("json");
    summary(&seq_len, output, export, query_length)?;
    Ok(())
}

impl CountFx {
    pub fn run(&self) -> Result<()> {
        set_up_threads(self.threads)?;
        count_fx(&self.fx, self.export, self.query_length)?;
        Ok(())
    }
}
