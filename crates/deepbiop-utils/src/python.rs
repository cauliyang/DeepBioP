use crate::blat;

use crate::{
    blat::PslAlignment,
    interval::{self, GenomicInterval, Overlap},
    strategy,
};

use ahash::HashMap;
use anyhow::Result;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::ops::Range;
use std::path::PathBuf;

use needletail::Sequence;

use pyo3_stub_gen::derive::*;

#[gen_stub_pymethods]
#[pymethods]
impl PslAlignment {
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "PslAlignment(qname={}, qsize={}, qstart={}, qend={}, qmatch={}, tname={}, tsize={}, tstart={}, tend={}, identity={})",
            self.qname, self.qsize, self.qstart, self.qend, self.qmatch, self.tname, self.tsize, self.tstart, self.tend, self.identity
        ))
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl GenomicInterval {
    #[new]
    fn py_new(chr: &str, start: usize, end: usize) -> Self {
        GenomicInterval {
            chr: chr.into(),
            start,
            end,
        }
    }

    #[getter]
    fn get_chr(&self) -> String {
        self.chr.to_string()
    }
    #[setter]
    fn set_chr(&mut self, chr: &str) {
        self.chr = chr.into();
    }

    #[pyo3(name = "overlap")]
    fn py_overlap(&self, other: &GenomicInterval) -> bool {
        self.overlap(other)
    }

    fn __repr__(&self) -> String {
        format!(
            "Segment(chr={}, start={}, end={})",
            self.chr, self.start, self.end
        )
    }
}

#[gen_stub_pyfunction(module = "deepbiop.utils")]
#[pyfunction]
fn majority_voting(labels: Vec<i8>, window_size: usize) -> Vec<i8> {
    strategy::majority_voting(&labels, window_size)
}

/// Parse PSL file by query name.
#[gen_stub_pyfunction(module = "deepbiop.utils")]
#[pyfunction]
fn parse_psl_by_qname(file_path: PathBuf) -> Result<HashMap<String, Vec<blat::PslAlignment>>> {
    blat::parse_psl_by_qname(file_path)
}

#[allow(clippy::type_complexity)]
#[gen_stub_pyfunction(module = "deepbiop.utils")]
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

#[gen_stub_pyfunction(module = "deepbiop.utils")]
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

#[gen_stub_pyfunction(module = "deepbiop.utils")]
#[pyfunction]
fn reverse_complement(seq: String) -> String {
    String::from_utf8(seq.as_bytes().reverse_complement()).unwrap()
}

// register utils module
pub fn register_utils_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let sub_module_name = "utils";
    let child_module = PyModule::new_bound(parent_module.py(), sub_module_name)?;

    child_module.add_class::<GenomicInterval>()?;
    child_module.add_class::<PslAlignment>()?;

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
    Ok(())
}
