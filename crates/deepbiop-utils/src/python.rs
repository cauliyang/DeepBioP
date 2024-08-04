use crate::blat;

use ahash::HashMap;
use anyhow::Result;
use pyo3::prelude::*;
use std::path::PathBuf;

#[pyfunction]
pub fn parse_psl_by_qname(file_path: PathBuf) -> Result<HashMap<String, Vec<blat::PslAlignment>>> {
    blat::parse_psl_by_qname(file_path)
}
