//! DataLoader class for PyTorch-style batch loading.
//!
//! This module provides the DataLoader class that wraps Dataset instances
//! and provides batching, shuffling, and parallel loading capabilities.

use pyo3::prelude::*;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

use super::dataset::Dataset;

/// PyTorch-compatible DataLoader for batching and shuffling.
///
/// Wraps Dataset to provide:
/// - __len__(): Returns number of batches
/// - __iter__(): Returns iterator over batches
/// - Batching with configurable batch_size
/// - Optional shuffling with seed control
#[pyclass(name = "DataLoader", module = "deepbiop.pytorch")]
pub struct DataLoader {
    /// Reference to the dataset
    dataset: Py<Dataset>,
    /// Batch size
    batch_size: usize,
    /// Whether to shuffle data
    shuffle: bool,
    /// Number of worker threads (currently unused, for future)
    num_workers: usize,
    /// Whether to drop last incomplete batch
    drop_last: bool,
    /// Random seed for shuffling
    seed: Option<u64>,
}

#[pymethods]
impl DataLoader {
    /// Create a new DataLoader.
    ///
    /// Args:
    ///     dataset: Dataset instance to load from
    ///     batch_size: Number of samples per batch (default: 1)
    ///     shuffle: Whether to shuffle data (default: False)
    ///     num_workers: Number of worker threads (default: 0, currently unused)
    ///     collate_fn: Function to collate samples into batch (default: None, currently unused)
    ///     drop_last: Drop last incomplete batch (default: False)
    ///     seed: Random seed for shuffling (default: None)
    ///
    /// Returns:
    ///     DataLoader instance
    #[new]
    #[pyo3(signature = (dataset, *, batch_size=1, shuffle=false, num_workers=0, collate_fn=None, drop_last=false, seed=None))]
    #[allow(unused_variables)]
    fn new(
        dataset: Py<Dataset>,
        batch_size: usize,
        shuffle: bool,
        num_workers: usize,
        collate_fn: Option<Py<PyAny>>,
        drop_last: bool,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        if batch_size == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "batch_size must be greater than 0",
            ));
        }

        // TODO: Handle collate_fn in future tasks

        Ok(DataLoader {
            dataset,
            batch_size,
            shuffle,
            num_workers,
            drop_last,
            seed,
        })
    }

    /// Returns number of batches.
    ///
    /// Calculated as:
    /// - If drop_last=False: ceil(dataset_size / batch_size)
    /// - If drop_last=True: floor(dataset_size / batch_size)
    fn __len__(&self, py: Python) -> PyResult<usize> {
        let dataset = self.dataset.borrow(py);
        let dataset_len = dataset.len();

        let num_batches = if self.drop_last {
            dataset_len / self.batch_size
        } else {
            dataset_len.div_ceil(self.batch_size)
        };

        Ok(num_batches)
    }

    /// Iterate over batches.
    ///
    /// Returns:
    ///     Iterator over batches (each batch is a list of samples)
    fn __iter__(slf: PyRef<'_, Self>) -> PyResult<DataLoaderIterator> {
        let py = slf.py();

        // Get dataset length without holding the borrow
        let dataset_len = {
            let dataset = slf.dataset.borrow(py);
            dataset.len()
        };

        // Create index array
        let mut indices: Vec<usize> = (0..dataset_len).collect();

        // Shuffle if requested
        if slf.shuffle {
            if let Some(seed) = slf.seed {
                let mut rng = StdRng::seed_from_u64(seed);
                indices.shuffle(&mut rng);
            } else {
                // Use rng() for non-seeded random shuffling
                let mut thread_rng = rand::rng();
                indices.shuffle(&mut thread_rng);
            }
        }

        Ok(DataLoaderIterator {
            dataloader: slf.into(),
            indices,
            current_batch: 0,
        })
    }

    /// Human-readable representation.
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "DataLoader(batch_size={}, shuffle={}, num_workers={}, drop_last={})",
            self.batch_size, self.shuffle, self.num_workers, self.drop_last
        ))
    }
}

/// Iterator for DataLoader.
///
/// Maintains iteration state and yields batches as lists of samples.
#[pyclass(name = "DataLoaderIterator", module = "deepbiop.pytorch")]
pub struct DataLoaderIterator {
    dataloader: Py<DataLoader>,
    indices: Vec<usize>,
    current_batch: usize,
}

#[pymethods]
impl DataLoaderIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python) -> PyResult<Option<Py<pyo3::types::PyList>>> {
        let dataloader = self.dataloader.borrow(py);
        let dataset = dataloader.dataset.borrow(py);

        let total_batches = if dataloader.drop_last {
            self.indices.len() / dataloader.batch_size
        } else {
            self.indices.len().div_ceil(dataloader.batch_size)
        };

        if self.current_batch >= total_batches {
            return Ok(None);
        }

        // Calculate batch boundaries
        let start_idx = self.current_batch * dataloader.batch_size;
        let end_idx = std::cmp::min(start_idx + dataloader.batch_size, self.indices.len());

        // Collect samples for this batch
        let batch_list = pyo3::types::PyList::empty(py);
        for &idx in &self.indices[start_idx..end_idx] {
            let sample = dataset.get_item(idx, py)?;
            batch_list.append(sample)?;
        }

        self.current_batch += 1;

        Ok(Some(batch_list.into()))
    }
}
