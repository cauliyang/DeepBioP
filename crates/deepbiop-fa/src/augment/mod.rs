//! Data augmentation operations for FASTA records.
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
//! use deepbiop_fa::augment::{ReverseComplement, Mutator, Augmentation};
//!
//! // Create augmenters
//! let mut rc = ReverseComplement::new();
//! let mut mutator = Mutator::new(0.01, Some(42)); // 1% mutation rate, seed 42
//!
//! // Apply augmentations
//! let sequence = b"ACGTACGT";
//! let rc_sequence = rc.apply(sequence);
//! ```

pub mod mutator;
pub mod reverse_complement;
pub mod sampler;

pub use mutator::{Mutator, MutatorBuilder};
pub use reverse_complement::ReverseComplement;
pub use sampler::cut_fa_randomly;

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
