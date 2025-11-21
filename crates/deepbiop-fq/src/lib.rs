//! **deepbiop-fq** is a Rust library for reading and writing FASTQ files.

pub mod augment;
pub mod dataset;
pub mod encode;
pub mod error;
pub mod filter;
pub mod io;
pub mod predicts;
pub mod streaming;
pub mod types;
pub mod utils;

#[cfg(feature = "python")]
pub mod python;
