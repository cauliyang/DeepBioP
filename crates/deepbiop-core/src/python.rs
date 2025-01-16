use crate::{kmer, seq, types};

use anyhow::Result;
use needletail::Sequence;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::*;
use rayon::prelude::*;

#[gen_stub_pyfunction(module = "deepbiop.core")]
#[pyfunction]
fn seq_to_kmers(seq: String, k: usize, overlap: bool) -> Vec<String> {
    let normalized_seq = seq.as_bytes().normalize(false);
    kmer::seq_to_kmers(&normalized_seq, k, overlap)
        .par_iter()
        .map(|s| String::from_utf8_lossy(s).to_string())
        .collect()
}

#[gen_stub_pyfunction(module = "deepbiop.core")]
#[pyfunction]
fn kmers_to_seq(kmers: Vec<String>) -> Result<String> {
    let kmers_as_bytes: Vec<&[u8]> = kmers.par_iter().map(|s| s.as_bytes()).collect();
    Ok(String::from_utf8_lossy(&kmer::kmers_to_seq(kmers_as_bytes)?).to_string())
}

#[gen_stub_pyfunction(module = "deepbiop.core")]
#[pyfunction]
fn generate_kmers_table(base: String, k: usize) -> types::Kmer2IdTable {
    let base = base.as_bytes();
    kmer::generate_kmers_table(base, k as u8)
}

#[gen_stub_pyfunction(module = "deepbiop.core")]
#[pyfunction]
fn generate_kmers(base: String, k: usize) -> Vec<String> {
    let base = base.as_bytes();
    kmer::generate_kmers(base, k as u8)
        .into_iter()
        .map(|s| String::from_utf8_lossy(&s).to_string())
        .collect()
}

// register fq sub_module
pub fn register_core_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let sub_module_name = "core";
    let child_module = PyModule::new(parent_module.py(), sub_module_name)?;

    child_module.add_function(wrap_pyfunction!(seq::normalize_seq, &child_module)?)?;
    child_module.add_function(wrap_pyfunction!(seq_to_kmers, &child_module)?)?;
    child_module.add_function(wrap_pyfunction!(kmers_to_seq, &child_module)?)?;
    child_module.add_function(wrap_pyfunction!(generate_kmers, &child_module)?)?;
    child_module.add_function(wrap_pyfunction!(generate_kmers_table, &child_module)?)?;

    parent_module.add_submodule(&child_module)?;

    Ok(())
}
