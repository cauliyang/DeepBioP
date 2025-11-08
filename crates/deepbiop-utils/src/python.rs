use crate::{blat, highlight_targets, interval::GenomicInterval, io};

use ahash::HashMap;
use pyo3::prelude::*;
use std::path::PathBuf;

/// Check the compression type of a file
#[pyfunction]
fn check_compressed_type(file_path: PathBuf) -> PyResult<io::CompressedType> {
    io::check_compressed_type(file_path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))
}

/// Check if a file is compressed
#[pyfunction]
fn is_compressed(file_path: PathBuf) -> PyResult<bool> {
    io::is_compressed(file_path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))
}

/// Check the sequence file type
#[pyfunction]
fn check_sequence_file_type(file_path: PathBuf) -> PyResult<io::SequenceFileType> {
    io::check_sequence_file_type(file_path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))
}

/// Parse PSL file and return alignments grouped by query name
#[pyfunction]
fn parse_psl_by_qname(file_path: PathBuf) -> PyResult<HashMap<String, Vec<blat::PslAlignment>>> {
    blat::parse_psl_by_qname(file_path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))
}

/// Parse PSL file and return all alignments
#[pyfunction]
fn parse_psl(file_path: PathBuf) -> PyResult<Vec<blat::PslAlignment>> {
    blat::parse_psl(file_path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))
}

/// Register the utils module with Python
pub fn register_utils_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let sub_module_name = "utils";
    let child_module = PyModule::new(parent_module.py(), sub_module_name)?;

    // Add classes
    child_module.add_class::<GenomicInterval>()?;
    child_module.add_class::<io::CompressedType>()?;
    child_module.add_class::<io::SequenceFileType>()?;
    child_module.add_class::<blat::PslAlignment>()?;

    // Add functions
    child_module.add_function(wrap_pyfunction!(highlight_targets, &child_module)?)?;
    child_module.add_function(wrap_pyfunction!(check_compressed_type, &child_module)?)?;
    child_module.add_function(wrap_pyfunction!(is_compressed, &child_module)?)?;
    child_module.add_function(wrap_pyfunction!(check_sequence_file_type, &child_module)?)?;
    child_module.add_function(wrap_pyfunction!(parse_psl_by_qname, &child_module)?)?;
    child_module.add_function(wrap_pyfunction!(parse_psl, &child_module)?)?;

    parent_module.add_submodule(&child_module)?;

    Ok(())
}
