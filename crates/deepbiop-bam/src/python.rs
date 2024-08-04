use anyhow::Result;
use noodles::sam::record::Cigar;
use pyo3::prelude::*;

use crate::cigar::calc_softclips;

#[pyfunction]
pub fn left_right_soft_clip(cigar_string: &str) -> Result<(usize, usize)> {
    let cigar = Cigar::new(cigar_string.as_bytes());
    calc_softclips(&cigar)
}

// register bam sub module
pub fn register_bam_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let child_module = PyModule::new_bound(parent_module.py(), "bam")?;
    child_module.add_function(wrap_pyfunction!(left_right_soft_clip, &child_module)?)?;

    parent_module.add_submodule(&child_module)?;
    Ok(())
}
