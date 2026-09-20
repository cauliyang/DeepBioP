//! Data augmentation operations for FASTQ records.
//!
//! This module provides functionality to augment biological sequence data
//! for machine learning training, including:
//! - Reverse complement transformation
//! - Random mutations (substitutions)
//! - Random subsequence extraction
//!
//! # Examples
//!
//! ```no_run
//! use deepbiop_fq::augment::{ReverseComplement, Mutator, Augmentation};
//!
//! // Create augmenters
//! let mut rc = ReverseComplement::new();
//! let mutator = Mutator::new(0.01, Some(42)); // 1% mutation rate, seed 42
//!
//! // Apply augmentations
//! let sequence = b"ACGTACGT";
//! let rc_sequence = rc.apply(sequence);
//! ```

pub mod mutator;
pub mod quality;
pub mod reverse_complement;
pub mod sampler;

#[cfg(feature = "python")]
pub mod python;

pub use mutator::{Mutator, MutatorBuilder};
pub use quality::{QualityModel, QualitySimulator, QualitySimulatorBuilder};
pub use reverse_complement::ReverseComplement;
pub use sampler::{Sampler, SamplerBuilder};

/// Trait for sequence augmentation operations.
pub trait Augmentation {
    /// Apply the augmentation to a sequence.
    ///
    /// # Arguments
    ///
    /// * `sequence` - The input sequence (bytes)
    ///
    /// # Returns
    ///
    /// The augmented sequence
    fn apply(&mut self, sequence: &[u8]) -> Vec<u8>;
}
