use needletail::Sequence;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::*;

/// Normalize a DNA sequence by converting any non-standard nucleotides to standard ones.
///
/// This function takes a DNA sequence as a `String` and a boolean flag `iupac` indicating whether to normalize using IUPAC ambiguity codes.
/// It returns a normalized DNA sequence as a `String`.
///
/// # Arguments
///
/// * `seq` - A DNA sequence as a `String`.
/// * `iupac` - A boolean flag indicating whether to normalize using IUPAC ambiguity codes.
///
/// # Returns
///
/// A normalized DNA sequence as a `String`.
#[gen_stub_pyfunction(module = "deepbiop.core")]
#[pyfunction]
pub fn normalize_seq(seq: String, iupac: bool) -> String {
    String::from_utf8_lossy(&seq.as_bytes().normalize(iupac)).to_string()
}
