//! Cache layer for processed datasets.
//!
//! This module provides caching functionality to speed up repeated
//! data loading by storing processed datasets to disk.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::fs;
use std::path::Path;

/// Save processed samples to cache file.
///
/// Saves samples to .npz format (NumPy compressed) with accompanying .meta.json metadata file
/// for cache invalidation based on source file modification time.
///
/// Args:
///     samples: List of processed samples (dicts with 'sequence' key as NumPy arrays)
///     cache_path: Path to save cache file (should end with .npz)
///     source_file: Optional source file path for staleness detection
///
/// Raises:
///     ValueError: If samples list is empty
///     IOError: If file write fails
///
/// Examples:
///     >>> save_cache(processed_samples, "cache.npz", source_file="data.fastq")
#[pyfunction]
#[pyo3(signature = (samples, cache_path, source_file=None))]
pub fn save_cache(
    py: Python,
    samples: &Bound<'_, PyList>,
    cache_path: String,
    source_file: Option<String>,
) -> PyResult<()> {
    // Import numpy
    let np = py.import("numpy")?;
    let json_module = py.import("json")?;
    let pathlib = py.import("pathlib")?;

    // Ensure cache_path ends with appropriate extension
    let cache_path_obj = pathlib.getattr("Path")?.call1((cache_path.clone(),))?;

    // Prepare data for saving
    // Extract sequences and other fields from samples
    let num_samples = samples.len();

    if num_samples == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Cannot save empty samples list to cache",
        ));
    }

    // Build dict of arrays for np.savez_compressed
    let save_dict = PyDict::new(py);

    // Collect all sequences
    let mut sequences = Vec::new();
    let mut has_quality = false;

    for (idx, sample) in samples.iter().enumerate() {
        let sample_dict = sample.downcast::<PyDict>()?;

        // Get sequence array
        if let Some(seq) = sample_dict.get_item("sequence")? {
            sequences.push(seq);
        } else {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Sample {} missing 'sequence' key",
                idx
            )));
        }

        // Check if quality exists (only need to check first sample)
        if idx == 0 {
            has_quality = sample_dict.contains("quality")?;
        }
    }

    // Save each sequence individually (to handle variable lengths)
    for (idx, seq) in sequences.iter().enumerate() {
        save_dict.set_item(format!("seq_{}", idx), seq)?;
    }

    save_dict.set_item("num_samples", num_samples)?;

    // Save metadata
    let metadata = PyDict::new(py);
    metadata.set_item("num_samples", num_samples)?;
    metadata.set_item("has_quality", has_quality)?;

    // Add source file info if provided
    if let Some(source) = &source_file {
        let source_path = Path::new(source);
        if source_path.exists() {
            let mtime = fs::metadata(source_path)
                .and_then(|m| m.modified())
                .map(|t| t.duration_since(std::time::UNIX_EPOCH).unwrap().as_secs())
                .unwrap_or(0);

            metadata.set_item("source_file", source)?;
            metadata.set_item("source_mtime", mtime)?;
        }
    }

    // Save arrays using numpy.savez_compressed
    let savez_compressed = np.getattr("savez_compressed")?;
    savez_compressed.call((cache_path.clone(),), Some(&save_dict))?;

    // Save metadata JSON (append .meta.json to full cache path)
    let meta_path = format!("{}.meta.json", cache_path);
    let meta_file = fs::File::create(&meta_path)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

    let metadata_json = json_module.getattr("dumps")?.call1((metadata,))?;
    let metadata_str: String = metadata_json.extract()?;

    std::io::Write::write_all(
        &mut std::io::BufWriter::new(meta_file),
        metadata_str.as_bytes(),
    )
    .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

    Ok(())
}

/// Load processed samples from cache file.
///
/// Args:
///     cache_path: Path to cache file (.npz)
///
/// Returns:
///     List of samples (dicts with 'sequence' key as NumPy arrays)
///
/// Raises:
///     FileNotFoundError: If cache file not found
///     IOError: If load fails or file is corrupted
///
/// Examples:
///     >>> samples = load_cache("cache.npz")
///     >>> len(samples)
///     1000
#[pyfunction]
pub fn load_cache(py: Python, cache_path: String) -> PyResult<Py<PyList>> {
    // Import numpy
    let np = py.import("numpy")?;

    // Check cache file exists
    if !Path::new(&cache_path).exists() {
        return Err(pyo3::exceptions::PyFileNotFoundError::new_err(format!(
            "Cache file not found: {}",
            cache_path
        )));
    }

    // Load arrays using numpy.load
    let load_fn = np.getattr("load")?;
    let loaded = load_fn.call1((cache_path,))?;

    // Extract data - loaded is a NpzFile object, access with indexing
    let num_samples_obj = loaded.get_item("num_samples")?;
    let num_samples: usize = num_samples_obj.extract()?;

    // Reconstruct samples list
    let samples = PyList::empty(py);

    for idx in 0..num_samples {
        let sample = PyDict::new(py);

        // Get sequence for this sample
        let seq_key = format!("seq_{}", idx);
        let seq = loaded.get_item(seq_key.as_str())?;
        sample.set_item("sequence", seq)?;

        samples.append(sample)?;
    }

    Ok(samples.into())
}

/// Check if cache is valid (not stale).
///
/// Validates cache by comparing source file modification time (mtime) with cached metadata.
/// Returns False if cache file missing, metadata missing, or source file has been modified.
///
/// Args:
///     cache_path: Path to cache file (.npz)
///     source_file: Source file path to check against (optional)
///
/// Returns:
///     True if cache is valid (source file unchanged), False otherwise
///
/// Examples:
///     >>> is_cache_valid("cache.npz", source_file="data.fastq")
///     True
#[pyfunction]
#[pyo3(signature = (cache_path, source_file=None))]
pub fn is_cache_valid(
    py: Python,
    cache_path: String,
    source_file: Option<String>,
) -> PyResult<bool> {
    // Check cache file exists
    if !Path::new(&cache_path).exists() {
        return Ok(false);
    }

    // Check metadata file exists (append .meta.json to full cache path)
    let meta_path = format!("{}.meta.json", cache_path);
    if !Path::new(&meta_path).exists() {
        return Ok(false);
    }

    // If no source file specified, cache is valid (can't check staleness)
    let source = match source_file {
        Some(s) => s,
        None => return Ok(true),
    };

    // Check source file exists
    let source_path = Path::new(&source);
    if !source_path.exists() {
        return Ok(false);
    }

    // Load metadata
    let json_module = py.import("json")?;
    let meta_content = fs::read_to_string(&meta_path)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

    let loads_fn = json_module.getattr("loads")?;
    let metadata = loads_fn.call1((meta_content,))?;
    let metadata_dict = metadata.downcast::<PyDict>()?;

    // Compare source file mtime
    if let Ok(Some(cached_source)) = metadata_dict.get_item("source_file") {
        let cached_source_str: String = cached_source.extract()?;

        // Check paths match
        if cached_source_str != source {
            return Ok(false);
        }

        // Check modification time
        if let Ok(Some(cached_mtime_obj)) = metadata_dict.get_item("source_mtime") {
            let cached_mtime: u64 = cached_mtime_obj.extract()?;

            let current_mtime = fs::metadata(source_path)
                .and_then(|m| m.modified())
                .map(|t| t.duration_since(std::time::UNIX_EPOCH).unwrap().as_secs())
                .unwrap_or(0);

            // Cache is valid if mtime hasn't changed
            return Ok(cached_mtime == current_mtime);
        }
    }

    // If no metadata about source file, assume cache is valid
    Ok(true)
}

/// Register cache functions with Python module.
pub fn register_cache_functions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(save_cache, m)?)?;
    m.add_function(wrap_pyfunction!(load_cache, m)?)?;
    m.add_function(wrap_pyfunction!(is_cache_valid, m)?)?;
    Ok(())
}
