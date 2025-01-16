//! **deepbiop-fq** is a Rust library for reading and writing FASTQ files.

pub mod encode;
pub mod error;
pub mod io;
pub mod predicts;
pub mod types;
pub mod utils;

#[cfg(feature = "python")]
pub mod python;
