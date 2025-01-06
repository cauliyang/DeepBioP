//! **deepbiop-fa** is a Rust library for reading and writing FASTA files.
pub mod encode;
pub mod io;

#[cfg(feature = "python")]
pub mod python;
