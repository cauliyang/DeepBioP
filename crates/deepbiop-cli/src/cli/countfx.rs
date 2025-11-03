use std::{
    fs::File,
    io::{BufReader, BufWriter, Write},
    path::{Path, PathBuf},
};

use bon::Builder;
use clap::Parser;
use deepbiop_utils as utils;
use noodles::{fasta, fastq};

use super::set_up_threads;
use anyhow::Result;
use rayon::prelude::*;

#[derive(Debug, Parser)]
pub struct CountFx {
    /// path to the fastx file
    #[arg(value_name = "fx", action=clap::ArgAction::Append)]
    fx: PathBuf,

    /// if export the result
    #[arg(long, action=clap::ArgAction::SetTrue)]
    export: bool,

    /// threads number
    #[arg(short, long, default_value = "2")]
    threads: Option<usize>,

    // query length
    #[arg(long, default_value = "0")]
    query_length: usize,
}

#[derive(Debug, Default, Builder)]
struct Statistics {
    sorted_lens: Vec<usize>,
    total_len: usize,
    max_len: usize,
    min_len: usize,
    mean_len: f64,
    std_dev: f64,
    q1: usize,
    q2: usize,
    q3: usize,
    p90: usize,
    p95: usize,
    p99: usize,
    query_length: usize,
    #[builder(default)]
    count_query_length: usize,
    #[builder(default)]
    percentage_query_length: f64,
}


fn human_readable_bases(bases: usize) -> String {
    const UNITS: [&str; 5] = ["", "K", "M", "G", "T"];
    if bases < 1000 {
        return format!("{} bp", bases);
    }
    let mut size = bases as f64;
    let mut unit = 0usize;
    // Find the largest unit without overshooting
    while size >= 1000.0 && unit < UNITS.len() - 1 {
        size /= 1000.0;
        unit += 1;
    }
    // Remove unnecessary decimal zeros (e.g., 1.00Kb -> 1Kb)
    let size_str = if (size * 100.0) % 100.0 == 0.0 {
        format!("{:.0}", size)
    } else if (size * 10.0) % 10.0 == 0.0 {
        format!("{:.1}", size)
    } else {
        format!("{:.2}", size)
    };
    format!("{}{}b", size_str, UNITS[unit])
}


impl Statistics {
    fn json<P: AsRef<Path>>(&self, output: P) -> Result<()> {
        let json_file = File::create(output)?;
        let mut json_writer = BufWriter::new(json_file);
        json_writer.write_all(serde_json::to_string(&self.sorted_lens)?.as_bytes())?;
        Ok(())
    }

    fn count_query_length(&mut self) {
        if self.query_length > 0 {
            match self.sorted_lens.binary_search(&self.query_length) {
                Ok(idx) | Err(idx) => {
                    let count = self.sorted_lens.len() - idx;

                    self.count_query_length = count;
                    self.percentage_query_length =
                        count as f64 / self.sorted_lens.len() as f64 * 100.0;
                }
            }
        }
    }
    fn print(&self) {
        println!(
            "The number of sequences                       : {}",
            self.sorted_lens.len()
        );

        println!(
            "The total bases in sequences                  : {} ({})",
            self.total_len,
            human_readable_bases(self.total_len)
        );

        println!(
            "The minimum length of sequences               : {}",
            self.min_len
        );
        println!(
            "The maximum length of sequences               : {}",
            self.max_len
        );
        println!(
            "The mean length of sequences                  : {:.2}",
            self.mean_len
        );
        println!(
            "The standard deviation of sequences           : {:.2}",
            self.std_dev
        );

        println!(
            "The first quartile of sequences               : {}",
            self.q1
        );
        println!(
            "The second quartile of sequences              : {}",
            self.q2
        );
        println!(
            "The third quartile of sequences               : {}",
            self.q3
        );

        println!(
            "The 90th percentile of sequences              : {}",
            self.p90
        );
        println!(
            "The 95th percentile of sequences              : {}",
            self.p95
        );
        println!(
            "The 99th percentile of sequences              : {}",
            self.p99
        );

        if self.query_length > 0 {
            println!(
                "The number of sequences with length >= {}     : {}",
                self.query_length, self.count_query_length
            );
            println!(
                "The percentage of sequences with length >= {} : {:.4}%",
                self.query_length, self.percentage_query_length
            );
        }
    }
}

fn summary<P: AsRef<Path>>(
    seq_len: &[usize],
    output: P,
    export: bool,
    query_length: usize,
) -> Result<()> {
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

    let mut statistics = Statistics::builder()
        .sorted_lens(sorted_lens)
        .total_len(total_len)
        .max_len(max_len)
        .min_len(min_len)
        .mean_len(mean_len)
        .std_dev(std_dev)
        .q1(q1)
        .q2(q2)
        .q3(q3)
        .p90(p90)
        .p95(p95)
        .p99(p99)
        .query_length(query_length)
        .build();

    statistics.count_query_length();
    statistics.print();

    if export {
        statistics.json(output)?;
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
