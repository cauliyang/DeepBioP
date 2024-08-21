//! **deepbiop-bam** is a library for working with BAM files.

pub mod chimeric;
pub mod cigar;
pub mod io;

#[cfg(feature = "python")]
pub mod python;
