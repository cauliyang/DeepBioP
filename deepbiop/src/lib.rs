#![warn(missing_docs)]

//! **DeepBiop** is a library for preprocessing bilogical data (fastq, fastq, bam .etc.) to apply
//! data for deep learning models.


#[cfg(feature = "fastq")]
#[doc(inline)]
pub use deepbiop_fq as fastq;
