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
    /// path to the fasta or fastq file
    #[arg(value_name = "fa")]
    fx: PathBuf,

    /// if export the result
    #[arg(long, action=clap::ArgAction::SetTrue)]
    export: bool,

    /// threads number
    #[arg(short, long, default_value = "2")]
    threads: Option<usize>,
}

fn export_json<P: AsRef<Path>>(seq_len: &[usize], output: P) -> Result<()> {
    // save identities to json file
    let json_file = File::create(output)?;
    let mut json_writer = BufWriter::new(json_file);
    json_writer.write_all(serde_json::to_string(seq_len)?.as_bytes())?;
    Ok(())
}

fn summary<P: AsRef<Path>>(seq_len: &[usize], output: P, export: bool) -> Result<()> {
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

    if export {
        export_json(seq_len, output)?;
    }
    Ok(())
}

fn count_fx<P: AsRef<Path>>(fx: P, export: bool) -> Result<()> {
    let reader = utils::io::create_reader_for_compressed_file(&fx)?;

    let seq_len = match utils::io::check_sequence_file_type(&fx)? {
        utils::io::SequenceFileType::Fasta => {
            let mut reader = fasta::Reader::new(BufReader::new(reader));
            reader
                .records()
                .map(|record| record.map(|r| r.sequence().len()))
                .collect::<Result<Vec<_>, _>>()?
        }
        utils::io::SequenceFileType::Fastq => {
            let mut reader = fastq::Reader::new(BufReader::new(reader));
            reader
                .records()
                .map(|record| record.map(|r| r.sequence().len()))
                .collect::<Result<Vec<_>, _>>()?
        }
        _ => return Err(anyhow::anyhow!("Unsupported file type")),
    };

    let output = fx.as_ref().with_extension("json");
    summary(&seq_len, output, export)?;
    Ok(())
}

impl CountFx {
    pub fn run(&self) -> Result<()> {
        set_up_threads(self.threads)?;
        count_fx(&self.fx, self.export)?;
        Ok(())
    }
}
