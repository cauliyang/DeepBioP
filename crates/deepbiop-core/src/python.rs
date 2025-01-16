use crate::{kmer, seq, types};

use anyhow::Result;
use needletail::Sequence;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::*;
use rayon::prelude::*;

/// Convert a DNA sequence into k-mers.
///
/// This function takes a DNA sequence and splits it into k-mers of specified length.
/// The sequence is first normalized to handle non-standard nucleotides.
///
/// # Arguments
///
/// * `seq` - A DNA sequence as a `String`
/// * `k` - The length of each k-mer
/// * `overlap` - Whether to generate overlapping k-mers
///
/// # Returns
///
/// A vector of k-mers as `String`s
#[gen_stub_pyfunction(module = "deepbiop.core")]
#[pyfunction]
fn seq_to_kmers(seq: String, k: usize, overlap: bool) -> Vec<String> {
    let normalized_seq = seq.as_bytes().normalize(false);
    kmer::seq_to_kmers(&normalized_seq, k, overlap)
        .par_iter()
        .map(|s| String::from_utf8_lossy(s).to_string())
        .collect()
}

/// Convert k-mers back into a DNA sequence.
///
/// This function takes a vector of k-mers and reconstructs the original DNA sequence.
/// The k-mers are assumed to be in order and overlapping.
///
/// # Arguments
///
/// * `kmers` - A vector of k-mers as `String`s
///
/// # Returns
///
/// The reconstructed DNA sequence as a `String`, wrapped in a `Result`
#[gen_stub_pyfunction(module = "deepbiop.core")]
#[pyfunction]
fn kmers_to_seq(kmers: Vec<String>) -> Result<String> {
    let kmers_as_bytes: Vec<&[u8]> = kmers.par_iter().map(|s| s.as_bytes()).collect();
    Ok(String::from_utf8_lossy(&kmer::kmers_to_seq(kmers_as_bytes)?).to_string())
}

/// Generate a lookup table mapping k-mers to unique IDs.
///
/// This function takes a string of base characters and a k-mer length,
/// and generates a HashMap mapping each possible k-mer to a unique integer ID.
///
/// # Arguments
///
/// * `base` - A string containing the base characters to use (e.g. "ATCG")
/// * `k` - The length of k-mers to generate
///
/// # Returns
///
/// A HashMap mapping k-mer byte sequences to integer IDs
#[gen_stub_pyfunction(module = "deepbiop.core")]
#[pyfunction]
fn generate_kmers_table(base: String, k: usize) -> types::Kmer2IdTable {
    let base = base.as_bytes();
    kmer::generate_kmers_table(base, k as u8)
}

/// Generate all possible k-mers from a set of base characters.
///
/// This function takes a string of base characters and a k-mer length,
/// and generates all possible k-mer combinations of that length.
///
/// # Arguments
///
/// * `base` - A string containing the base characters to use (e.g. "ATCG")
/// * `k` - The length of k-mers to generate
///
/// # Returns
///
/// A vector containing all possible k-mer combinations as strings
#[gen_stub_pyfunction(module = "deepbiop.core")]
#[pyfunction]
fn generate_kmers(base: String, k: usize) -> Vec<String> {
    let base = base.as_bytes();
    kmer::generate_kmers(base, k as u8)
        .into_iter()
        .map(|s| String::from_utf8_lossy(&s).to_string())
        .collect()
}

// register python sub_module
pub fn register_core_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let sub_module_name = "core";
    let child_module = PyModule::new(parent_module.py(), sub_module_name)?;

    child_module.add_function(wrap_pyfunction!(seq::normalize_seq, &child_module)?)?;
    child_module.add_function(wrap_pyfunction!(seq_to_kmers, &child_module)?)?;
    child_module.add_function(wrap_pyfunction!(kmers_to_seq, &child_module)?)?;
    child_module.add_function(wrap_pyfunction!(generate_kmers, &child_module)?)?;
    child_module.add_function(wrap_pyfunction!(generate_kmers_table, &child_module)?)?;
    child_module.add_function(wrap_pyfunction!(seq::reverse_complement, &child_module)?)?;

    parent_module.add_submodule(&child_module)?;

    Ok(())
}
