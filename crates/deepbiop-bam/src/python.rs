use std::path::PathBuf;

use ahash::HashMap;
use anyhow::Result;
use noodles::sam::record::Cigar;
use pyo3::prelude::*;

use crate::chimeric;
use crate::cigar::calc_softclips;

#[pyfunction]
fn count_chimeric_reads_for_path(bam: PathBuf, threads: Option<usize>) -> Result<usize> {
    chimeric::count_chimeric_reads_for_path(bam, threads)
}

#[pyfunction]
fn count_chimeric_reads_for_paths(
    bams: Vec<PathBuf>,
    threads: Option<usize>,
) -> Result<HashMap<PathBuf, usize>> {
    Ok(chimeric::count_chimeric_reads_for_paths(&bams, threads))
}

#[pyfunction]
fn left_right_soft_clip(cigar_string: &str) -> Result<(usize, usize)> {
    let cigar = Cigar::new(cigar_string.as_bytes());
    calc_softclips(&cigar)
}

// register bam sub module
pub fn register_bam_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let child_module = PyModule::new_bound(parent_module.py(), "bam")?;
    child_module.add_function(wrap_pyfunction!(left_right_soft_clip, &child_module)?)?;
    child_module.add_function(wrap_pyfunction!(
        count_chimeric_reads_for_path,
        &child_module
    )?)?;
    child_module.add_function(wrap_pyfunction!(
        count_chimeric_reads_for_paths,
        &child_module
    )?)?;

    parent_module.add_submodule(&child_module)?;
    Ok(())
}
