//! **deepbiop-fa** is a Rust library for reading and writing FASTA files.
pub mod augment;
pub mod dataset;
pub mod encode;
pub mod io;

#[cfg(feature = "python")]
pub mod python;
