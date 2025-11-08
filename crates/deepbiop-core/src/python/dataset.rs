//! Python bindings for dataset and batch functionality.

use numpy::{PyArray2, ToPyArray};
use pyo3::prelude::*;
use pyo3_stub_gen::derive::*;

use crate::batch::{Batch, BatchBuilder, PaddingStrategy};
use crate::seq::SequenceRecord;

/// Python wrapper for Batch struct.
///
/// This class represents a batched collection of biological sequences ready for GPU processing.
/// It handles padding, masking, and collation of variable-length sequences.
///
/// # Attributes
///
/// * `ids` - List of sequence identifiers
/// * `sequences` - 2D numpy array [batch_size, max_length] containing sequences
/// * `quality_scores` - Optional 2D numpy array [batch_size, max_length] containing quality scores
/// * `attention_mask` - 2D numpy array [batch_size, max_length] (1=real, 0=padding)
/// * `lengths` - List of original sequence lengths before padding
#[gen_stub_pyclass(module = "deepbiop.core")]
#[pyclass(name = "Batch")]
pub struct PyBatch {
    batch: Batch,
}

#[pymethods]
impl PyBatch {
    /// Get the sequence IDs.
    ///
    /// # Returns
    ///
    /// List of sequence identifier strings
    #[getter]
    fn ids(&self) -> Vec<String> {
        self.batch.ids.clone()
    }

    /// Get the padded sequences as a numpy array.
    ///
    /// # Returns
    ///
    /// 2D numpy array with shape [batch_size, max_length]
    #[getter]
    fn sequences<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<u8>> {
        self.batch.sequences.to_pyarray(py)
    }

    /// Get the quality scores as a numpy array (if available).
    ///
    /// # Returns
    ///
    /// Optional 2D numpy array with shape [batch_size, max_length], or None
    #[getter]
    fn quality_scores<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<u8>>> {
        self.batch.quality_scores.as_ref().map(|q| q.to_pyarray(py))
    }

    /// Get the attention mask as a numpy array.
    ///
    /// Values are 1 for real sequence data, 0 for padding.
    ///
    /// # Returns
    ///
    /// 2D numpy array with shape [batch_size, max_length]
    #[getter]
    fn attention_mask<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<u8>> {
        self.batch.attention_mask.to_pyarray(py)
    }

    /// Get the original sequence lengths before padding.
    ///
    /// # Returns
    ///
    /// List of sequence lengths
    #[getter]
    fn lengths(&self) -> Vec<usize> {
        self.batch.lengths.clone()
    }

    /// Get the batch size (number of sequences).
    ///
    /// # Returns
    ///
    /// Number of sequences in this batch
    #[pyo3(name = "__len__")]
    fn len(&self) -> usize {
        self.batch.size()
    }

    /// Get the maximum sequence length in this batch.
    ///
    /// # Returns
    ///
    /// Maximum length after padding
    fn max_length(&self) -> usize {
        self.batch.max_length()
    }

    /// String representation of the batch.
    #[pyo3(name = "__repr__")]
    fn repr(&self) -> String {
        format!(
            "Batch(size={}, max_length={}, has_quality={})",
            self.batch.size(),
            self.batch.max_length(),
            self.batch.quality_scores.is_some()
        )
    }
}

/// Collate a list of SequenceRecords into a Batch.
///
/// This function handles padding and masking of variable-length sequences
/// for efficient GPU processing.
///
/// # Arguments
///
/// * `records` - List of SequenceRecord objects to batch
/// * `padding` - Padding strategy: "longest", "fixed", or "bucketed"
/// * `max_length` - Maximum sequence length (for "fixed" padding only)
/// * `pad_value` - Value to use for padding (default: 0)
/// * `truncate` - Whether to truncate sequences exceeding max_length (default: False)
///
/// # Returns
///
/// A Batch object containing padded sequences and metadata
///
/// # Examples
///
/// ```python
/// from deepbiop.core import SequenceRecord, collate_batch
///
/// records = [
///     SequenceRecord("seq1", b"ACGT", None, None),
///     SequenceRecord("seq2", b"TG", None, None),
/// ]
///
/// # Dynamic padding to longest sequence
/// batch = collate_batch(records, padding="longest")
///
/// # Fixed-length padding
/// batch = collate_batch(records, padding="fixed", max_length=10)
/// ```
#[gen_stub_pyfunction(module = "deepbiop.core")]
#[pyfunction]
#[pyo3(signature = (records, padding="longest", max_length=None, pad_value=0, truncate=false))]
fn collate_batch(
    records: Vec<SequenceRecord>,
    padding: &str,
    max_length: Option<usize>,
    pad_value: u8,
    truncate: bool,
) -> PyResult<PyBatch> {
    // Parse padding strategy
    let strategy = match padding {
        "longest" => PaddingStrategy::Longest,
        "fixed" => {
            let len = max_length.ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(
                    "max_length must be specified for fixed padding",
                )
            })?;
            PaddingStrategy::Fixed { length: len }
        }
        "bucketed" => {
            // For now, use default buckets. Can be extended to accept custom boundaries.
            PaddingStrategy::Bucketed {
                boundaries: vec![100, 200, 500, 1000],
            }
        }
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown padding strategy: {}. Use 'longest', 'fixed', or 'bucketed'",
                padding
            )))
        }
    };

    // Build the batch
    let batch = BatchBuilder::new()
        .padding_strategy(strategy)
        .pad_value(pad_value)
        .truncate(truncate)
        .build(&records)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    Ok(PyBatch { batch })
}

/// Register dataset-related functions and classes with the Python module.
pub(crate) fn register_dataset_functions(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyBatch>()?;
    module.add_function(wrap_pyfunction!(collate_batch, module)?)?;
    Ok(())
}
