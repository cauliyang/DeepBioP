//! Default values for the configuration of the Deepbiop.

// Quality score offset used in FASTQ files (Phred+33)
pub const QUAL_OFFSET: u8 = 33;

// Standard DNA bases used for k-mer generation (includes N for ambiguous bases)
pub const BASES: &[u8] = b"ATCGN";

// Default k-mer size for sequence analysis
pub const KMER_SIZE: u8 = 3;

// Flag to control whether targets should be vectorized
pub const VECTORIZED_TARGET: bool = false;

// Special label value used to indicate entries that should be ignored during processing
pub const IGNORE_LABEL: i64 = -100;
