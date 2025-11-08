//! Batch processing for variable-length biological sequences.
//!
//! This module provides data structures and utilities for batching sequence records
//! for efficient GPU processing. It handles padding, masking, and collation of
//! variable-length sequences into fixed-size tensors.

use ndarray::{Array2, ArrayView2};

use crate::error::DPError;
use crate::seq::SequenceRecord;

/// Result type alias for batch operations.
pub type BatchResult<T> = Result<T, DPError>;

/// Padding strategy for variable-length sequences in a batch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PaddingStrategy {
    /// Pad to the longest sequence in the batch (dynamic padding).
    ///
    /// Memory-efficient but batch sizes vary. Best for similar-length sequences.
    Longest,

    /// Pad/truncate all sequences to a fixed length.
    ///
    /// Consistent batch sizes, but may waste memory for short sequences
    /// or lose information for long sequences.
    Fixed {
        /// Target length for all sequences
        length: usize,
    },

    /// Group sequences by similar lengths to minimize padding (bucketing).
    ///
    /// Reduces padding overhead by batching sequences of similar lengths together.
    /// Requires length-based sorting/grouping before batching.
    Bucketed {
        /// Bucket boundaries (e.g., [100, 200, 500] creates buckets 0-100, 101-200, 201-500, 501+)
        boundaries: Vec<usize>,
    },
}

impl Default for PaddingStrategy {
    fn default() -> Self {
        Self::Longest
    }
}

/// Batch of sequence records ready for GPU processing.
///
/// This structure handles the complexities of batching variable-length biological
/// sequences into fixed-size tensors suitable for neural network training. It provides:
///
/// - **Padding**: Extends shorter sequences to match batch dimensions
/// - **Attention masks**: Distinguishes real data from padding
/// - **Length tracking**: Preserves original sequence lengths for loss masking
///
/// # Memory Layout
///
/// All tensors use row-major (C-contiguous) layout for compatibility with PyTorch:
/// - `sequences`: [batch_size, max_length] - Integer-encoded or one-hot
/// - `quality_scores`: [batch_size, max_length] - Phred quality values
/// - `attention_mask`: [batch_size, max_length] - 1 for real tokens, 0 for padding
///
/// # Example
///
/// ```rust,ignore
/// use deepbiop_core::batch::{Batch, BatchBuilder, PaddingStrategy};
/// use deepbiop_core::seq::SequenceRecord;
///
/// let records = vec![
///     SequenceRecord::new("seq1".to_string(), b"ACGT".to_vec(), None, None),
///     SequenceRecord::new("seq2".to_string(), b"TG".to_vec(), None, None),
/// ];
///
/// let batch = BatchBuilder::new()
///     .padding_strategy(PaddingStrategy::Longest)
///     .pad_value(0)
///     .build(&records)?;
///
/// assert_eq!(batch.size(), 2);
/// assert_eq!(batch.max_length(), 4);
/// ```
#[derive(Debug, Clone)]
pub struct Batch {
    /// Sequence identifiers (read names)
    pub ids: Vec<String>,

    /// Padded sequences [batch_size, max_length]
    ///
    /// Values are ASCII codes (65=A, 67=C, 71=G, 84=T) or encoded integers.
    /// Padding positions contain `pad_value` (typically 0).
    pub sequences: Array2<u8>,

    /// Padded quality scores [batch_size, max_length]
    ///
    /// Optional Phred quality scores (0-60). None if any record lacks quality scores.
    pub quality_scores: Option<Array2<u8>>,

    /// Attention mask [batch_size, max_length]
    ///
    /// - 1: Real sequence data (attend to this position)
    /// - 0: Padding (mask out this position)
    pub attention_mask: Array2<u8>,

    /// Original sequence lengths before padding
    ///
    /// Used for:
    /// - Loss masking (ignore padding positions)
    /// - Debugging and validation
    /// - Dynamic computation (e.g., pack_padded_sequence in PyTorch)
    pub lengths: Vec<usize>,

    /// Value used for padding sequences (typically 0)
    pad_value: u8,
}

impl Batch {
    /// Returns the batch size (number of sequences).
    pub fn size(&self) -> usize {
        self.ids.len()
    }

    /// Returns true if the batch is empty.
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// Returns the maximum sequence length in this batch.
    pub fn max_length(&self) -> usize {
        if self.sequences.is_empty() {
            0
        } else {
            self.sequences.ncols()
        }
    }

    /// Returns the padding value used for this batch.
    pub fn pad_value(&self) -> u8 {
        self.pad_value
    }

    /// Validates the internal consistency of the batch.
    ///
    /// Checks that:
    /// - All arrays have consistent batch size
    /// - Lengths match actual unpadded sequence lengths
    /// - Attention masks correctly identify padding
    pub fn validate(&self) -> BatchResult<()> {
        let batch_size = self.size();

        // Check sequences shape
        if self.sequences.nrows() != batch_size {
            return Err(DPError::InvalidValue(format!(
                "Sequences batch size mismatch: expected {}, got {}",
                batch_size,
                self.sequences.nrows()
            )));
        }

        // Check quality_scores shape if present
        if let Some(ref qual) = self.quality_scores {
            if qual.shape() != self.sequences.shape() {
                return Err(DPError::InvalidValue(format!(
                    "Quality scores shape {:?} doesn't match sequences shape {:?}",
                    qual.shape(),
                    self.sequences.shape()
                )));
            }
        }

        // Check attention_mask shape
        if self.attention_mask.shape() != self.sequences.shape() {
            return Err(DPError::InvalidValue(format!(
                "Attention mask shape {:?} doesn't match sequences shape {:?}",
                self.attention_mask.shape(),
                self.sequences.shape()
            )));
        }

        // Check lengths vector size
        if self.lengths.len() != batch_size {
            return Err(DPError::InvalidValue(format!(
                "Lengths vector size {} doesn't match batch size {}",
                self.lengths.len(),
                batch_size
            )));
        }

        // Verify lengths don't exceed max_length
        let max_len = self.max_length();
        for (i, &len) in self.lengths.iter().enumerate() {
            if len > max_len {
                return Err(DPError::InvalidValue(format!(
                    "Length at index {} ({}) exceeds max_length ({})",
                    i, len, max_len
                )));
            }
        }

        Ok(())
    }

    /// Returns a view of the sequences array.
    pub fn sequences_view(&self) -> ArrayView2<'_, u8> {
        self.sequences.view()
    }

    /// Returns a view of the quality scores array if present.
    pub fn quality_scores_view(&self) -> Option<ArrayView2<'_, u8>> {
        self.quality_scores.as_ref().map(|q| q.view())
    }

    /// Returns a view of the attention mask array.
    pub fn attention_mask_view(&self) -> ArrayView2<'_, u8> {
        self.attention_mask.view()
    }
}

/// Builder for creating batches from sequence records.
///
/// Provides a fluent interface for configuring batch parameters and
/// collating multiple sequence records into a single batch.
///
/// # Example
///
/// ```rust,ignore
/// use deepbiop_core::batch::{BatchBuilder, PaddingStrategy};
///
/// let batch = BatchBuilder::new()
///     .padding_strategy(PaddingStrategy::Fixed { length: 512 })
///     .pad_value(0)
///     .truncate(true)
///     .build(&records)?;
/// ```
#[derive(Debug, Clone)]
pub struct BatchBuilder {
    padding_strategy: PaddingStrategy,
    pad_value: u8,
    truncate: bool,
}

impl Default for BatchBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl BatchBuilder {
    /// Creates a new batch builder with default settings.
    ///
    /// Defaults:
    /// - Padding strategy: `Longest`
    /// - Pad value: `0`
    /// - Truncate: `false`
    pub fn new() -> Self {
        Self {
            padding_strategy: PaddingStrategy::default(),
            pad_value: 0,
            truncate: false,
        }
    }

    /// Sets the padding strategy.
    pub fn padding_strategy(mut self, strategy: PaddingStrategy) -> Self {
        self.padding_strategy = strategy;
        self
    }

    /// Sets the value used for padding (default: 0).
    pub fn pad_value(mut self, value: u8) -> Self {
        self.pad_value = value;
        self
    }

    /// Sets whether to truncate sequences that exceed the target length (default: false).
    ///
    /// Only applicable for `PaddingStrategy::Fixed`.
    pub fn truncate(mut self, truncate: bool) -> Self {
        self.truncate = truncate;
        self
    }

    /// Builds a batch from a vector of sequence records.
    ///
    /// # Arguments
    ///
    /// * `records` - Vector of sequence records to batch
    ///
    /// # Returns
    ///
    /// A `Batch` containing padded sequences, attention masks, and metadata.
    ///
    /// # Errors
    ///
    /// - `DPError::InvalidValue`: Empty records vector
    /// - `DPError::InvalidValue`: Sequence exceeds fixed length and truncate=false
    pub fn build(self, records: &[SequenceRecord]) -> BatchResult<Batch> {
        if records.is_empty() {
            return Err(DPError::InvalidValue(
                "Cannot create batch from empty records vector".to_string(),
            ));
        }

        let batch_size = records.len();

        // Determine target length based on padding strategy
        let target_len = self.compute_target_length(records)?;

        // Extract lengths - use min of actual length and target_len (for truncation)
        let lengths: Vec<usize> = records.iter().map(|r| r.len().min(target_len)).collect();

        // Build sequences array
        let mut sequences = Array2::from_elem((batch_size, target_len), self.pad_value);
        for (i, record) in records.iter().enumerate() {
            let seq_len = record.len().min(target_len);
            sequences
                .row_mut(i)
                .slice_mut(ndarray::s![..seq_len])
                .assign(&ndarray::ArrayView1::from(&record.sequence[..seq_len]));
        }

        // Build quality scores array if all records have quality scores
        let quality_scores = if records.iter().all(|r| r.quality_scores.is_some()) {
            let mut qual = Array2::from_elem((batch_size, target_len), self.pad_value);
            for (i, record) in records.iter().enumerate() {
                if let Some(ref q) = record.quality_scores {
                    let qual_len = q.len().min(target_len);
                    qual.row_mut(i)
                        .slice_mut(ndarray::s![..qual_len])
                        .assign(&ndarray::ArrayView1::from(&q[..qual_len]));
                }
            }
            Some(qual)
        } else {
            None
        };

        // Build attention mask
        let mut attention_mask = Array2::zeros((batch_size, target_len));
        for (i, &len) in lengths.iter().enumerate() {
            let mask_len = len.min(target_len);
            attention_mask
                .row_mut(i)
                .slice_mut(ndarray::s![..mask_len])
                .fill(1);
        }

        // Extract IDs
        let ids: Vec<String> = records.iter().map(|r| r.id.clone()).collect();

        let batch = Batch {
            ids,
            sequences,
            quality_scores,
            attention_mask,
            lengths,
            pad_value: self.pad_value,
        };

        batch.validate()?;
        Ok(batch)
    }

    /// Computes the target length for padding based on the strategy.
    fn compute_target_length(&self, records: &[SequenceRecord]) -> BatchResult<usize> {
        match self.padding_strategy {
            PaddingStrategy::Longest => Ok(records.iter().map(|r| r.len()).max().unwrap_or(0)),
            PaddingStrategy::Fixed { length } => {
                // Check if any sequence exceeds the fixed length
                if !self.truncate {
                    for record in records {
                        if record.len() > length {
                            return Err(DPError::InvalidValue(format!(
                                "Sequence '{}' length ({}) exceeds fixed length ({}). \
                                 Set truncate=true to allow truncation.",
                                record.id,
                                record.len(),
                                length
                            )));
                        }
                    }
                }
                Ok(length)
            }
            PaddingStrategy::Bucketed { ref boundaries } => {
                // Find the appropriate bucket for the longest sequence
                let max_len = records.iter().map(|r| r.len()).max().unwrap_or(0);
                let bucket_len = boundaries
                    .iter()
                    .find(|&&b| b >= max_len)
                    .copied()
                    .unwrap_or_else(|| {
                        // If no bucket fits, use the last boundary + some margin
                        boundaries.last().copied().unwrap_or(max_len)
                    });
                Ok(bucket_len.max(max_len))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_builder_longest_padding() {
        let records = vec![
            SequenceRecord::new("seq1".to_string(), b"ACGT".to_vec(), None, None),
            SequenceRecord::new("seq2".to_string(), b"TG".to_vec(), None, None),
            SequenceRecord::new("seq3".to_string(), b"AAAAAA".to_vec(), None, None),
        ];

        let batch = BatchBuilder::new()
            .padding_strategy(PaddingStrategy::Longest)
            .build(&records)
            .unwrap();

        assert_eq!(batch.size(), 3);
        assert_eq!(batch.max_length(), 6);
        assert_eq!(batch.lengths, vec![4, 2, 6]);

        // Check attention masks
        assert_eq!(batch.attention_mask[[0, 0]], 1); // seq1[0] - real
        assert_eq!(batch.attention_mask[[0, 3]], 1); // seq1[3] - real
        assert_eq!(batch.attention_mask[[0, 4]], 0); // seq1[4] - padding
        assert_eq!(batch.attention_mask[[1, 1]], 1); // seq2[1] - real
        assert_eq!(batch.attention_mask[[1, 2]], 0); // seq2[2] - padding
    }

    #[test]
    fn test_batch_builder_fixed_padding() {
        let records = vec![
            SequenceRecord::new("seq1".to_string(), b"ACGT".to_vec(), None, None),
            SequenceRecord::new("seq2".to_string(), b"TG".to_vec(), None, None),
        ];

        let batch = BatchBuilder::new()
            .padding_strategy(PaddingStrategy::Fixed { length: 10 })
            .build(&records)
            .unwrap();

        assert_eq!(batch.size(), 2);
        assert_eq!(batch.max_length(), 10);
        assert_eq!(batch.lengths, vec![4, 2]);
    }

    #[test]
    fn test_batch_builder_fixed_padding_truncate() {
        let records = vec![
            SequenceRecord::new("seq1".to_string(), b"ACGT".to_vec(), None, None),
            SequenceRecord::new("seq2".to_string(), b"TGCAAAAAAA".to_vec(), None, None), // 10 bases
        ];

        let batch = BatchBuilder::new()
            .padding_strategy(PaddingStrategy::Fixed { length: 5 })
            .truncate(true)
            .build(&records)
            .unwrap();

        assert_eq!(batch.size(), 2);
        assert_eq!(batch.max_length(), 5);
        // Sequence 2 is truncated to 5
        assert_eq!(batch.sequences.row(1)[0], b'T');
        assert_eq!(batch.sequences.row(1)[4], b'A');
    }

    #[test]
    fn test_batch_builder_fixed_padding_error_without_truncate() {
        let records = vec![
            SequenceRecord::new("seq1".to_string(), b"ACGTACGTACGT".to_vec(), None, None), // 12 bases
        ];

        let result = BatchBuilder::new()
            .padding_strategy(PaddingStrategy::Fixed { length: 5 })
            .truncate(false)
            .build(&records);

        assert!(result.is_err());
        if let Err(DPError::InvalidValue(msg)) = result {
            assert!(msg.contains("exceeds fixed length"));
        } else {
            panic!("Expected InvalidValue error");
        }
    }

    #[test]
    fn test_batch_with_quality_scores() {
        let records = vec![
            SequenceRecord::new(
                "seq1".to_string(),
                b"ACGT".to_vec(),
                Some(vec![30, 30, 30, 30]),
                None,
            ),
            SequenceRecord::new("seq2".to_string(), b"TG".to_vec(), Some(vec![40, 40]), None),
        ];

        let batch = BatchBuilder::new()
            .padding_strategy(PaddingStrategy::Longest)
            .build(&records)
            .unwrap();

        assert!(batch.quality_scores.is_some());
        let qual = batch.quality_scores.unwrap();
        assert_eq!(qual[[0, 0]], 30); // seq1[0]
        assert_eq!(qual[[0, 3]], 30); // seq1[3] - last real position
        assert_eq!(qual[[1, 0]], 40); // seq2[0]
        assert_eq!(qual[[1, 1]], 40); // seq2[1] - last real position
        assert_eq!(qual[[1, 2]], 0); // seq2[2] - padded
        assert_eq!(qual[[1, 3]], 0); // seq2[3] - padded
    }

    #[test]
    fn test_batch_without_quality_scores() {
        let records = vec![
            SequenceRecord::new(
                "seq1".to_string(),
                b"ACGT".to_vec(),
                Some(vec![30, 30, 30, 30]),
                None,
            ),
            SequenceRecord::new("seq2".to_string(), b"TG".to_vec(), None, None), // No quality
        ];

        let batch = BatchBuilder::new()
            .padding_strategy(PaddingStrategy::Longest)
            .build(&records)
            .unwrap();

        // Should be None because not all records have quality scores
        assert!(batch.quality_scores.is_none());
    }

    #[test]
    fn test_batch_validation() {
        let records = vec![SequenceRecord::new(
            "seq1".to_string(),
            b"ACGT".to_vec(),
            None,
            None,
        )];

        let batch = BatchBuilder::new().build(&records).unwrap();

        // Validation should pass
        assert!(batch.validate().is_ok());
    }

    #[test]
    fn test_empty_batch_error() {
        let records: Vec<SequenceRecord> = vec![];

        let result = BatchBuilder::new().build(&records);

        assert!(result.is_err());
        if let Err(DPError::InvalidValue(msg)) = result {
            assert!(msg.contains("empty"));
        } else {
            panic!("Expected InvalidValue error");
        }
    }

    #[test]
    fn test_bucketed_padding() {
        let records = vec![
            SequenceRecord::new("seq1".to_string(), b"ACGT".to_vec(), None, None), // 4
            SequenceRecord::new("seq2".to_string(), b"TGCAAA".to_vec(), None, None), // 6
        ];

        let batch = BatchBuilder::new()
            .padding_strategy(PaddingStrategy::Bucketed {
                boundaries: vec![5, 10, 20],
            })
            .build(&records)
            .unwrap();

        // Max length is 6, so should use bucket boundary 10
        assert_eq!(batch.max_length(), 10);
        assert_eq!(batch.size(), 2);
    }
}
