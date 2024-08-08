use crate::blat;

use crate::{
    interval::{self},
    strategy,
};

use ahash::HashMap;
use anyhow::Result;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::ops::Range;
use std::path::PathBuf;

use needletail::Sequence;

#[pyclass]
#[derive(Debug)]
pub struct GenomicInterval2 {
    #[pyo3(get, set)]
    pub chr: String,
    #[pyo3(get, set)]
    pub start: usize,
    #[pyo3(get, set)]
    pub end: usize,
}

#[pymethods]
impl GenomicInterval2 {
    #[new]
    pub fn new(chr: String, start: usize, end: usize) -> Self {
        Self { chr, start, end }
    }
}

#[pyfunction]
fn majority_voting(labels: Vec<i8>, window_size: usize) -> Vec<i8> {
    strategy::majority_voting(&labels, window_size)
}

#[pyfunction]
fn parse_psl_by_qname(file_path: PathBuf) -> Result<HashMap<String, Vec<blat::PslAlignment>>> {
    blat::parse_psl_by_qname(file_path)
}

#[allow(clippy::type_complexity)]
#[pyfunction]
fn remove_intervals_and_keep_left(
    seq: String,
    intervals: Vec<(usize, usize)>,
) -> Result<(Vec<String>, Vec<(usize, usize)>)> {
    let intervals: Vec<Range<usize>> = intervals
        .par_iter()
        .map(|(start, end)| *start..*end)
        .collect();

    let (seqs, intevals) = interval::remove_intervals_and_keep_left(seq.as_bytes(), &intervals)?;
    Ok((
        seqs.par_iter().map(|s| s.to_string()).collect(),
        intevals.par_iter().map(|r| (r.start, r.end)).collect(),
    ))
}

#[pyfunction]
fn generate_unmaped_intervals(
    input: Vec<(usize, usize)>,
    total_length: usize,
) -> Vec<(usize, usize)> {
    let ranges: Vec<Range<usize>> = input.par_iter().map(|(start, end)| *start..*end).collect();
    interval::generate_unmaped_intervals(&ranges, total_length)
        .par_iter()
        .map(|r| (r.start, r.end))
        .collect()
}

#[pyfunction]
fn reverse_complement(seq: String) -> String {
    String::from_utf8(seq.as_bytes().reverse_complement()).unwrap()
}

// register utils module
pub fn register_utils_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let sub_module_name = "deepbiop.utils";
    let child_module = PyModule::new_bound(parent_module.py(), sub_module_name)?;

    child_module.add_class::<interval::GenomicInterval>()?;
    child_module.add_class::<blat::PslAlignment>()?;
    child_module.add_class::<GenomicInterval2>()?;

    child_module.add_function(wrap_pyfunction!(majority_voting, &child_module)?)?;
    child_module.add_function(wrap_pyfunction!(reverse_complement, &child_module)?)?;
    child_module.add_function(wrap_pyfunction!(crate::highlight_targets, &child_module)?)?;
    child_module.add_function(wrap_pyfunction!(parse_psl_by_qname, &child_module)?)?;
    child_module.add_function(wrap_pyfunction!(
        remove_intervals_and_keep_left,
        &child_module
    )?)?;
    child_module.add_function(wrap_pyfunction!(generate_unmaped_intervals, &child_module)?)?;

    parent_module.add_submodule(&child_module)?;

    parent_module
        .py()
        .import_bound("sys")?
        .getattr("modules")?
        .set_item(sub_module_name, &child_module)?;
    Ok(())
}
