#![warn(missing_docs)]

//! **DeepBiop** is a library for processing bilogical data (fastq, fastq, bam .etc.) to apply
//! data for deep learning models. It is written in Rust and provides a Python and Rust interface.

#[cfg(feature = "fastq")]
#[doc(inline)]
pub use deepbiop_fq as fastq;

#[cfg(feature = "bam")]
#[doc(inline)]
pub use deepbiop_bam as bam;

#[cfg(feature = "utils")]
#[doc(inline)]
pub use deepbiop_utils as utils;
