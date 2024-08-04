use anyhow::Result;
use noodles::sam::record::Cigar;
use pyo3::prelude::*;

use crate::cigar::calc_softclips;

#[pyfunction]
pub fn left_right_soft_clip(cigar_string: &str) -> Result<(usize, usize)> {
    let cigar = Cigar::new(cigar_string.as_bytes());
    calc_softclips(&cigar)
}
