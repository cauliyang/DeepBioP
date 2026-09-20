//! DataLoader class for PyTorch-style batch loading.
//!
//! This module provides the DataLoader class that wraps Dataset instances
//! and provides batching, shuffling, and parallel loading capabilities.

use pyo3::prelude::*;
use pyo3::types::PyList;
use pyo3_stub_gen::derive::*;
use rand::rngs::SmallRng;
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
#[gen_stub_pyclass]
#[pyclass(name = "DataLoader", module = "deepbiop.pytorch")]
pub struct DataLoader {
    /// Reference to the dataset
    dataset: Py<Dataset>,
    /// Batch size
    batch_size: usize,
    /// Whether to shuffle data
    shuffle: bool,
    /// Optional callable applied to each batch (list of samples)
    collate_fn: Option<Py<PyAny>>,
    /// Whether to drop last incomplete batch
    drop_last: bool,
    /// Random seed for shuffling
    seed: Option<u64>,
}

#[gen_stub_pymethods]
#[pymethods]
impl DataLoader {
    /// Create a new DataLoader.
    ///
    /// Args:
    ///     dataset: Dataset instance to load from
    ///     batch_size: Number of samples per batch (default: 1)
    ///     shuffle: Whether to shuffle data (default: False)
    ///     collate_fn: Callable receiving the list of samples of a batch and
    ///         returning the batch object (e.g. `deepbiop.pytorch.default_collate`).
    ///         When None, each batch is the plain list of samples.
    ///     drop_last: Drop last incomplete batch (default: False)
    ///     seed: Random seed for shuffling (default: None)
    ///
    /// This loader runs in-process. For multi-process loading use
    /// `torch.utils.data.DataLoader` on top of the dataset instead.
    ///
    /// Returns:
    ///     DataLoader instance
    #[new]
    #[pyo3(signature = (dataset, *, batch_size=1, shuffle=false, collate_fn=None, drop_last=false, seed=None))]
    fn new(
        dataset: Py<Dataset>,
        batch_size: usize,
        shuffle: bool,
        collate_fn: Option<Py<PyAny>>,
        drop_last: bool,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        if batch_size == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "batch_size must be greater than 0",
            ));
        }

        Ok(DataLoader {
            dataset,
            batch_size,
            shuffle,
            collate_fn,
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
    ///     Iterator over batches (each batch is a list of samples, or the
    ///     result of `collate_fn` applied to that list)
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
                let mut rng = SmallRng::seed_from_u64(seed);
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
            "DataLoader(batch_size={}, shuffle={}, collate_fn={}, drop_last={})",
            self.batch_size,
            self.shuffle,
            if self.collate_fn.is_some() {
                "set"
            } else {
                "None"
            },
            self.drop_last
        ))
    }
}

/// Iterator for DataLoader.
///
/// Maintains iteration state and yields batches as lists of samples.
#[gen_stub_pyclass]
#[pyclass(name = "DataLoaderIterator", module = "deepbiop.pytorch")]
pub struct DataLoaderIterator {
    dataloader: Py<DataLoader>,
    indices: Vec<usize>,
    current_batch: usize,
}

#[gen_stub_pymethods]
#[pymethods]
impl DataLoaderIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python) -> PyResult<Option<Py<PyAny>>> {
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
        let batch_list = PyList::empty(py);
        for &idx in &self.indices[start_idx..end_idx] {
            batch_list.append(dataset.get_item(idx, py)?)?;
        }
        self.current_batch += 1;

        match &dataloader.collate_fn {
            Some(collate) => collate.call1(py, (batch_list,)).map(Some),
            None => Ok(Some(batch_list.into_any().unbind())),
        }
    }
}
