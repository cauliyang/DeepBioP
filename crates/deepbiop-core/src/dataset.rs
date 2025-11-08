//! Dataset abstractions for machine learning pipelines.
//!
//! This module provides trait definitions for datasets that can be used with
//! PyTorch, Lightning, and other ML frameworks. It supports both streaming
//! (iterable) and random-access (map-style) datasets.

use std::path::PathBuf;

use crate::error::DPError;
use crate::seq::SequenceRecord;

/// Result type alias for dataset operations.
pub type DatasetResult<T> = Result<T, DPError>;

/// Iterable dataset trait for streaming large biological files.
///
/// This trait is designed for sequential access to datasets, particularly useful
/// for large files that don't fit in memory. It provides an iterator over sequence
/// records without requiring random access.
///
/// # Design Philosophy
///
/// - **Streaming-first**: Optimized for large files (>1GB) where loading all data
///   into memory is impractical
/// - **Zero-copy where possible**: Minimizes memory allocations during iteration
/// - **Composable**: Can be wrapped with transformations, caching, and batching
/// - **PyTorch compatible**: Designed to integrate with `torch.utils.data.IterableDataset`
///
/// # Example
///
/// ```rust,ignore
/// use deepbiop_core::dataset::IterableDataset;
///
/// struct FastqDataset {
///     path: PathBuf,
/// }
///
/// impl IterableDataset for FastqDataset {
///     fn iter(&self) -> Box<dyn Iterator<Item = DatasetResult<SequenceRecord>> + '_> {
///         // Return iterator over FASTQ records
///         todo!()
///     }
///
///     fn paths(&self) -> Vec<PathBuf> {
///         vec![self.path.clone()]
///     }
///
///     fn size_hint(&self) -> Option<usize> {
///         None  // Unknown until we parse the file
///     }
/// }
/// ```
///
/// # Thread Safety
///
/// Implementations must be `Send` to support PyTorch's multi-worker DataLoader.
/// The iterator itself doesn't need to be `Send` as each worker creates its own.
pub trait IterableDataset: Send {
    /// Returns an iterator over sequence records.
    ///
    /// Each call to `iter()` creates a fresh iterator that starts from the beginning
    /// of the dataset. This allows multiple independent iterations over the data.
    ///
    /// # Errors
    ///
    /// Iterator items are `Result<SequenceRecord, DPError>` to handle:
    /// - I/O errors during file reading
    /// - Parse errors for malformed records
    /// - Validation errors for invalid sequences
    ///
    /// Implementations should decide whether to:
    /// - Fail fast: Stop iteration on first error
    /// - Skip invalid: Log warning and continue with next record
    ///
    /// # Performance
    ///
    /// Implementations should consider:
    /// - Buffering reads to reduce I/O syscalls
    /// - Parallel decompression for .gz files
    /// - Prefetching next records while processing current
    fn iter(&self) -> Box<dyn Iterator<Item = DatasetResult<SequenceRecord>> + '_>;

    /// Returns the file paths associated with this dataset.
    ///
    /// Used for:
    /// - Cache key generation (file path + mtime)
    /// - Distributed training file splitting
    /// - Logging and debugging
    ///
    /// # Returns
    ///
    /// Vector of all file paths that this dataset reads from.
    /// Empty vector if dataset is in-memory or doesn't use files.
    fn paths(&self) -> Vec<PathBuf>;

    /// Returns a hint about the number of records in the dataset.
    ///
    /// # Returns
    ///
    /// - `Some(n)`: Dataset contains approximately n records
    /// - `None`: Size unknown (e.g., compressed files, streams)
    ///
    /// # Notes
    ///
    /// - This is a **hint**, not a guarantee. Actual count may differ due to:
    ///   - Filtering transformations that skip records
    ///   - Corrupted records that are skipped during iteration
    ///   - Estimation errors for compressed files
    /// - Used for progress bars, memory pre-allocation, and logging
    /// - If unsure, return `None` rather than an inaccurate estimate
    fn size_hint(&self) -> Option<usize>;
}

/// Map-style dataset trait for random access to biological data.
///
/// This trait is designed for datasets that support indexed access, allowing
/// retrieval of individual records by position. Best suited for:
/// - Small to medium datasets that fit in memory
/// - Pre-indexed file formats (BAM with .bai)
/// - Cached/preprocessed datasets
///
/// # Design Philosophy
///
/// - **Random access**: Support shuffling and batch sampling
/// - **PyTorch compatible**: Designed to integrate with `torch.utils.data.Dataset`
/// - **Consistent indexing**: `get(i)` always returns the same record
///
/// # Example
///
/// ```rust,ignore
/// use deepbiop_core::dataset::MapDataset;
///
/// struct InMemoryDataset {
///     records: Vec<SequenceRecord>,
/// }
///
/// impl MapDataset for InMemoryDataset {
///     fn len(&self) -> usize {
///         self.records.len()
///     }
///
///     fn get(&self, index: usize) -> DatasetResult<SequenceRecord> {
///         self.records.get(index)
///             .cloned()
///             .ok_or_else(|| DPError::InvalidParameter(
///                 format!("Index {} out of bounds", index)
///             ))
///     }
/// }
/// ```
///
/// # Thread Safety
///
/// Implementations must be `Send + Sync` to support:
/// - PyTorch's multi-worker DataLoader (Send)
/// - Concurrent reads from multiple workers (Sync)
pub trait MapDataset: Send + Sync {
    /// Returns the number of records in the dataset.
    ///
    /// This must return the exact count. Valid indices are `0..self.len()`.
    fn len(&self) -> usize;

    /// Returns true if the dataset is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Retrieves the record at the specified index.
    ///
    /// # Arguments
    ///
    /// * `index` - Zero-based index of the record to retrieve
    ///
    /// # Returns
    ///
    /// - `Ok(record)`: Successfully retrieved record at index
    /// - `Err(DPError::InvalidParameter)`: Index out of bounds
    /// - `Err(DPError::...)`: Other errors (e.g., I/O, decompression)
    ///
    /// # Performance
    ///
    /// Implementations should aim for O(1) or O(log n) access time.
    /// For file-backed datasets, consider:
    /// - Building an index during initialization
    /// - Caching recently accessed records
    /// - Using memory-mapped files for large datasets
    fn get(&self, index: usize) -> DatasetResult<SequenceRecord>;
}

#[cfg(test)]
mod tests {
    use super::*;

    // Simple in-memory iterable dataset for testing
    struct TestIterableDataset {
        records: Vec<SequenceRecord>,
        path: PathBuf,
    }

    impl IterableDataset for TestIterableDataset {
        fn iter(&self) -> Box<dyn Iterator<Item = DatasetResult<SequenceRecord>> + '_> {
            Box::new(self.records.iter().cloned().map(Ok))
        }

        fn paths(&self) -> Vec<PathBuf> {
            vec![self.path.clone()]
        }

        fn size_hint(&self) -> Option<usize> {
            Some(self.records.len())
        }
    }

    // Simple in-memory map dataset for testing
    struct TestMapDataset {
        records: Vec<SequenceRecord>,
    }

    impl MapDataset for TestMapDataset {
        fn len(&self) -> usize {
            self.records.len()
        }

        fn get(&self, index: usize) -> DatasetResult<SequenceRecord> {
            self.records.get(index).cloned().ok_or_else(|| {
                DPError::InvalidParameter(format!(
                    "Index {} out of bounds (len={})",
                    index,
                    self.len()
                ))
            })
        }
    }

    #[test]
    fn test_iterable_dataset_trait() {
        let records = vec![
            SequenceRecord::new("seq1".to_string(), b"ACGT".to_vec(), None, None),
            SequenceRecord::new("seq2".to_string(), b"TGCA".to_vec(), None, None),
        ];

        let dataset = TestIterableDataset {
            records: records.clone(),
            path: PathBuf::from("test.fastq"),
        };

        // Test iteration
        let collected: Vec<_> = dataset.iter().collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(collected.len(), 2);
        assert_eq!(collected[0].id, "seq1");
        assert_eq!(collected[1].id, "seq2");

        // Test size_hint
        assert_eq!(dataset.size_hint(), Some(2));

        // Test paths
        assert_eq!(dataset.paths(), vec![PathBuf::from("test.fastq")]);
    }

    #[test]
    fn test_map_dataset_trait() {
        let records = vec![
            SequenceRecord::new("seq1".to_string(), b"ACGT".to_vec(), None, None),
            SequenceRecord::new("seq2".to_string(), b"TGCA".to_vec(), None, None),
            SequenceRecord::new("seq3".to_string(), b"GGCC".to_vec(), None, None),
        ];

        let dataset = TestMapDataset {
            records: records.clone(),
        };

        // Test len
        assert_eq!(dataset.len(), 3);
        assert!(!dataset.is_empty());

        // Test get
        let rec0 = dataset.get(0).unwrap();
        assert_eq!(rec0.id, "seq1");
        assert_eq!(rec0.sequence, b"ACGT");

        let rec2 = dataset.get(2).unwrap();
        assert_eq!(rec2.id, "seq3");

        // Test out of bounds
        let result = dataset.get(10);
        assert!(result.is_err());
        if let Err(DPError::InvalidParameter(msg)) = result {
            assert!(msg.contains("out of bounds"));
        } else {
            panic!("Expected InvalidParameter error");
        }
    }

    #[test]
    fn test_empty_map_dataset() {
        let dataset = TestMapDataset { records: vec![] };

        assert_eq!(dataset.len(), 0);
        assert!(dataset.is_empty());

        let result = dataset.get(0);
        assert!(result.is_err());
    }

    #[test]
    fn test_multiple_iterations() {
        let records = vec![
            SequenceRecord::new("seq1".to_string(), b"ACGT".to_vec(), None, None),
            SequenceRecord::new("seq2".to_string(), b"TGCA".to_vec(), None, None),
        ];

        let dataset = TestIterableDataset {
            records: records.clone(),
            path: PathBuf::from("test.fastq"),
        };

        // First iteration
        let count1 = dataset.iter().count();
        assert_eq!(count1, 2);

        // Second iteration should work independently
        let count2 = dataset.iter().count();
        assert_eq!(count2, 2);

        // Verify data is unchanged
        let collected: Vec<_> = dataset.iter().collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(collected[0].id, "seq1");
    }
}
