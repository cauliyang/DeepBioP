//! Error conversion utilities for PyTorch-style API.
//!
//! This module provides conversion from DeepBioP Rust errors to Python exceptions,
//! ensuring user-friendly error messages with appropriate context.

use pyo3::exceptions::{PyFileNotFoundError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;

// Error conversion functions will be added in subsequent tasks
// Examples to be implemented:
// - Convert DPError to Python exceptions with context
// - Add file path and line number information
// - Provide actionable error messages for common issues

/// Convert a generic error to PyRuntimeError with context.
#[allow(dead_code)]
pub fn to_py_runtime_error<E: std::fmt::Display>(err: E) -> PyErr {
    PyErr::new::<PyRuntimeError, _>(format!("{}", err))
}

/// Convert a file not found error with path context.
#[allow(dead_code)]
pub fn to_py_file_not_found<S: AsRef<str>>(path: S, err: impl std::fmt::Display) -> PyErr {
    PyErr::new::<PyFileNotFoundError, _>(format!(
        "FASTQ/FASTA file '{}' not found. Check path and permissions. Error: {}",
        path.as_ref(),
        err
    ))
}

/// Convert a validation error with context.
#[allow(dead_code)]
pub fn to_py_value_error<S: AsRef<str>>(context: S, err: impl std::fmt::Display) -> PyErr {
    PyErr::new::<PyValueError, _>(format!("{}: {}", context.as_ref(), err))
}
