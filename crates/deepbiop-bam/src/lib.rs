//! **deepbiop-bam** is a library for working with BAM files.

pub mod chimeric;
pub mod cigar;
pub mod features;
pub mod io;
pub mod reader;

#[cfg(feature = "python")]
pub mod python;

pub use features::AlignmentFeatures;
pub use reader::BamReader;
