//! Cache layer for processed datasets.
//!
//! Processed samples are stored as a NumPy `.npz` archive (one entry per
//! sample key) plus a `.meta.json` sidecar recording the sample keys and the
//! source file's size/mtime for staleness checks.

use pyo3::prelude::*;
use pyo3_stub_gen::derive::*;
use pyo3::types::{PyBytes, PyDict, PyList};
use std::fs;
use std::io::Write;
use std::path::Path;

/// Cache path as numpy will actually write it: `np.savez` appends `.npz` when missing.
fn npz_path(cache_path: &str) -> String {
    if cache_path.ends_with(".npz") {
        cache_path.to_owned()
    } else {
        format!("{cache_path}.npz")
    }
}

fn meta_path(cache_path: &str) -> String {
    format!("{}.meta.json", npz_path(cache_path))
}

/// `(size, mtime_seconds)` of a file; `None` if it cannot be stat'ed.
fn source_signature(path: &Path) -> Option<(u64, u64)> {
    let meta = fs::metadata(path).ok()?;
    let mtime = meta
        .modified()
        .ok()?
        .duration_since(std::time::UNIX_EPOCH)
        .ok()?
        .as_secs();
    Some((meta.len(), mtime))
}

/// Save processed samples to a cache file.
///
/// Every key of every sample dict is stored (NumPy arrays, bytes, numbers,
/// strings), so a cached sample loads back with the same keys it was saved with.
///
/// Args:
///     samples: List of sample dicts; all samples must share the same keys
///     cache_path: Path to the cache file (`.npz` is appended if missing)
///     source_file: Optional source file path for staleness detection
///
/// Raises:
///     ValueError: If samples list is empty or samples have differing keys
///     IOError: If file write fails
///
/// Examples:
///     >>> save_cache(processed_samples, "cache.npz", source_file="data.fastq")
#[gen_stub_pyfunction(module = "deepbiop.pytorch")]
#[pyfunction]
#[pyo3(signature = (samples, cache_path, source_file=None))]
pub fn save_cache(
    py: Python,
    samples: &Bound<'_, PyList>,
    cache_path: String,
    source_file: Option<String>,
) -> PyResult<()> {
    let np = py.import("numpy")?;
    let json = py.import("json")?;

    let num_samples = samples.len();
    if num_samples == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Cannot save empty samples list to cache",
        ));
    }

    let first = samples.get_item(0)?.cast_into::<PyDict>()?;
    let mut keys: Vec<String> = first
        .keys()
        .iter()
        .map(|k| k.extract::<String>())
        .collect::<PyResult<_>>()?;
    keys.sort();
    if !keys.iter().any(|k| k == "sequence") {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Samples must have a 'sequence' key",
        ));
    }

    let save_dict = PyDict::new(py);
    for (idx, sample) in samples.iter().enumerate() {
        let sample = sample.cast::<PyDict>()?;
        if sample.len() != keys.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Sample {idx} has keys {:?}, expected {keys:?}",
                sample.keys()
            )));
        }
        for key in &keys {
            let value = sample.get_item(key)?.ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "Sample {idx} missing '{key}' key present in sample 0"
                ))
            })?;
            save_dict.set_item(format!("{key}_{idx}"), value)?;
        }
    }

    let metadata = PyDict::new(py);
    metadata.set_item("num_samples", num_samples)?;
    metadata.set_item("keys", &keys)?;
    if let Some(source) = &source_file {
        if let Some((size, mtime)) = source_signature(Path::new(source)) {
            metadata.set_item("source_file", source)?;
            metadata.set_item("source_size", size)?;
            metadata.set_item("source_mtime", mtime)?;
        }
    }

    np.getattr("savez_compressed")?
        .call((npz_path(&cache_path),), Some(&save_dict))?;

    let metadata_str: String = json.getattr("dumps")?.call1((metadata,))?.extract()?;
    let mut meta_file = fs::File::create(meta_path(&cache_path))
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    meta_file
        .write_all(metadata_str.as_bytes())
        .and_then(|_| meta_file.sync_all())
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

    Ok(())
}

/// Load processed samples from a cache file written by [`save_cache`].
///
/// Args:
///     cache_path: Path to cache file (`.npz`)
///
/// Returns:
///     List of sample dicts with the keys they were saved with; bytes values
///     come back as `bytes`, everything else as NumPy arrays/scalars.
///
/// Raises:
///     FileNotFoundError: If cache or metadata file not found
///     IOError: If load fails or file is corrupted
///
/// Examples:
///     >>> samples = load_cache("cache.npz")
///     >>> len(samples)
///     1000
#[gen_stub_pyfunction(module = "deepbiop.pytorch")]
#[pyfunction]
pub fn load_cache(py: Python, cache_path: String) -> PyResult<Py<PyList>> {
    let np = py.import("numpy")?;
    let json = py.import("json")?;

    let npz = npz_path(&cache_path);
    let meta = meta_path(&cache_path);
    for p in [&npz, &meta] {
        if !Path::new(p).exists() {
            return Err(pyo3::exceptions::PyFileNotFoundError::new_err(format!(
                "Cache file not found: {p}"
            )));
        }
    }

    let meta_content = fs::read_to_string(&meta)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    let metadata = json.getattr("loads")?.call1((meta_content,))?;
    let metadata = metadata.cast::<PyDict>()?;
    let num_samples: usize = metadata
        .get_item("num_samples")?
        .ok_or_else(|| pyo3::exceptions::PyIOError::new_err("cache metadata missing num_samples"))?
        .extract()?;
    let keys: Vec<String> = metadata
        .get_item("keys")?
        .ok_or_else(|| pyo3::exceptions::PyIOError::new_err("cache metadata missing keys"))?
        .extract()?;

    let loaded = np.getattr("load")?.call1((npz,))?;
    let samples = PyList::empty(py);
    for idx in 0..num_samples {
        let sample = PyDict::new(py);
        for key in &keys {
            let value = loaded.get_item(format!("{key}_{idx}"))?;
            // Bytes were stored as 0-d `S` arrays; restore them as bytes.
            let kind: String = value.getattr("dtype")?.getattr("kind")?.extract()?;
            if kind == "S" && value.getattr("ndim")?.extract::<usize>()? == 0 {
                let raw: Vec<u8> = value.call_method0("tobytes")?.extract()?;
                sample.set_item(key, PyBytes::new(py, &raw))?;
            } else {
                sample.set_item(key, value)?;
            }
        }
        samples.append(sample)?;
    }

    Ok(samples.into())
}

/// Check if a cache is present and not stale.
///
/// Returns False if the cache or its metadata is missing, or if `source_file`
/// is given and its size or modification time differ from what was recorded
/// when the cache was written.
///
/// Args:
///     cache_path: Path to cache file (`.npz`)
///     source_file: Source file path to check against (optional)
///
/// Examples:
///     >>> is_cache_valid("cache.npz", source_file="data.fastq")
///     True
#[gen_stub_pyfunction(module = "deepbiop.pytorch")]
#[pyfunction]
#[pyo3(signature = (cache_path, source_file=None))]
pub fn is_cache_valid(
    py: Python,
    cache_path: String,
    source_file: Option<String>,
) -> PyResult<bool> {
    let meta = meta_path(&cache_path);
    if !Path::new(&npz_path(&cache_path)).exists() || !Path::new(&meta).exists() {
        return Ok(false);
    }

    let Some(source) = source_file else {
        return Ok(true);
    };
    let Some((size, mtime)) = source_signature(Path::new(&source)) else {
        return Ok(false);
    };

    let meta_content = fs::read_to_string(&meta)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    let metadata = py
        .import("json")?
        .getattr("loads")?
        .call1((meta_content,))?;
    let metadata = metadata.cast::<PyDict>()?;

    let Some(cached_source) = metadata.get_item("source_file")? else {
        // Cache was written without a source file: nothing to compare against.
        return Ok(true);
    };
    if cached_source.extract::<String>()? != source {
        return Ok(false);
    }
    let cached_size: u64 = match metadata.get_item("source_size")? {
        Some(v) => v.extract()?,
        None => return Ok(false),
    };
    let cached_mtime: u64 = match metadata.get_item("source_mtime")? {
        Some(v) => v.extract()?,
        None => return Ok(false),
    };
    Ok(cached_size == size && cached_mtime == mtime)
}

/// Register cache functions with Python module.
pub fn register_cache_functions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(save_cache, m)?)?;
    m.add_function(wrap_pyfunction!(load_cache, m)?)?;
    m.add_function(wrap_pyfunction!(is_cache_valid, m)?)?;
    Ok(())
}
