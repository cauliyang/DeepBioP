use std::{
    io::BufReader,
    path::{Path, PathBuf},
};

use clap::Parser;
use deepbiop_utils as utils;
use noodles::fasta;

use super::set_up_threads;
use anyhow::Result;
use rayon::prelude::*;

#[derive(Debug, Parser)]
pub struct CountFa {
    /// path to the bam file
    #[arg(value_name = "fa")]
    fa: PathBuf,

    /// threads number
    #[arg(short, long, default_value = "2")]
    threads: Option<usize>,
}

fn summary(seq_len: &[usize]) -> Result<()> {
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

    let q1 = sorted_lens[q1_idx];
    let q2 = sorted_lens[q2_idx]; // median
    let q3 = sorted_lens[q3_idx];

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

    println!("The number of sequences: {}", seq_len.len());
    println!("The minimum length of sequences: {}", min_len);
    println!("The maximum length of sequences: {}", max_len);
    println!("The mean length of sequences: {:.2}", mean_len);
    println!("The standard deviation of sequences: {:.2}", std_dev);

    println!("The first quartile of sequences: {}", q1);
    println!("The second quartile of sequences: {}", q2);
    println!("The third quartile of sequences: {}", q3);
    Ok(())
}

fn count_fa<P: AsRef<Path>>(fa: P) -> Result<()> {
    let reader = utils::io::create_reader(fa)?;
    let mut reader = fasta::Reader::new(BufReader::new(reader));

    let seq_len: Vec<usize> = reader
        .records()
        .map(|record| record.map(|r| r.sequence().len()))
        .collect::<Result<Vec<_>, _>>()?;

    summary(&seq_len)?;
    Ok(())
}

impl CountFa {
    pub fn run(&self) -> Result<()> {
        set_up_threads(self.threads)?;
        count_fa(&self.fa)?;
        Ok(())
    }
}
