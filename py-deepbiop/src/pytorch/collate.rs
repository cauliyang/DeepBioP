//! Collate functions for batching variable-length sequences.
//!
//! This module provides functions for combining samples into batches,
//! with support for padding and stacking operations.

use numpy::{PyArray2, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

/// Default collate function for batching variable-length sequences.
///
/// Takes a list of samples (each with 'sequence' as 2D NumPy array) and:
/// 1. Finds maximum sequence length in the batch
/// 2. Pads all sequences to max_length with zeros
/// 3. Stacks into batch array [batch_size, max_len, features]
/// 4. Records original lengths for masking/packing
///
/// Args:
///     samples: List of sample dicts, each with 'sequence' key (2D NumPy float32 array)
///
/// Returns:
///     Batch dict with:
///     - 'sequences': NumPy array of shape [batch_size, max_len, features]
///     - 'lengths': NumPy array of original sequence lengths (1D int32)
///     - Other sample keys preserved as lists (e.g., 'quality', 'metadata')
///
/// Raises:
///     ValueError: If samples list is empty, missing 'sequence' key, or sequences have inconsistent feature dimensions
///
/// Examples:
///     >>> batch = default_collate([sample1, sample2, sample3])
///     >>> batch['sequences'].shape
///     (3, 150, 4)  # 3 samples, padded to 150bp, 4 features (one-hot DNA)
#[pyfunction]
pub fn default_collate(py: Python, samples: &Bound<'_, PyList>) -> PyResult<Py<PyDict>> {
    if samples.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Cannot collate empty batch",
        ));
    }

    // Convert samples to vector of dicts
    let sample_dicts: Vec<Bound<'_, PyDict>> = samples
        .iter()
        .map(|s| s.downcast_into::<PyDict>())
        .collect::<Result<Vec<_>, _>>()?;

    if sample_dicts.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Cannot collate empty batch",
        ));
    }

    // Extract sequences and find max length
    let mut sequences = Vec::new();
    let mut lengths = Vec::new();
    let mut max_len = 0;
    let mut feature_dim = 0;

    for sample in &sample_dicts {
        let sequence = sample.get_item("sequence")?.ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("Sample missing 'sequence' key")
        })?;

        // Get sequence as NumPy array
        let seq_array = sequence.downcast_into::<numpy::PyArrayDyn<f32>>()?;
        let shape = seq_array.shape();

        if shape.len() != 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Expected 2D sequence array, got {} dimensions",
                shape.len()
            )));
        }

        let seq_len = shape[0];
        let feat_dim = shape[1];

        if feature_dim == 0 {
            feature_dim = feat_dim;
        } else if feat_dim != feature_dim {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "All sequences must have same feature dimension",
            ));
        }

        max_len = max_len.max(seq_len);
        lengths.push(seq_len);
        sequences.push(seq_array);
    }

    // Create padded batch array
    let batch_size = sequences.len();
    let batch_array =
        unsafe { PyArray2::<f32>::new(py, [batch_size, max_len * feature_dim], false) };

    // Fill batch array with padded sequences
    unsafe {
        let batch_slice = batch_array.as_slice_mut()?;

        for (batch_idx, seq) in sequences.iter().enumerate() {
            let seq_data = seq.readonly();
            let seq_slice = seq_data.as_slice()?;
            let seq_len = lengths[batch_idx] * feature_dim;

            // Copy sequence data
            let batch_offset = batch_idx * max_len * feature_dim;
            batch_slice[batch_offset..batch_offset + seq_len]
                .copy_from_slice(&seq_slice[..seq_len]);

            // Zero-pad remaining
            for i in seq_len..max_len * feature_dim {
                batch_slice[batch_offset + i] = 0.0;
            }
        }
    }

    // Reshape to [batch_size, max_len, feature_dim]
    let batch_reshaped = batch_array.reshape([batch_size, max_len, feature_dim])?;

    // Create output dict
    let batch_dict = PyDict::new(py);
    batch_dict.set_item("sequences", batch_reshaped)?;

    // Add lengths array
    let lengths_array = numpy::PyArray::from_vec(py, lengths);
    batch_dict.set_item("lengths", lengths_array)?;

    // Copy other keys as lists
    if let Some(first_sample) = sample_dicts.first() {
        for (key, _) in first_sample.iter() {
            let key_str: String = key.extract()?;
            if key_str != "sequence" {
                // Collect values for this key from all samples
                let values = PyList::empty(py);
                for sample in &sample_dicts {
                    if let Some(value) = sample.get_item(&*key_str)? {
                        values.append(value)?;
                    }
                }
                batch_dict.set_item(key_str, values)?;
            }
        }
    }

    Ok(batch_dict.into())
}
