//! # deepbiop-bam
//! deepbiop-bam is a library for working with BAM files.

pub mod chimeric;
pub mod cigar;

#[cfg(feature = "python")]
pub mod python;
