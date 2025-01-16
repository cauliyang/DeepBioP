//! Python module for deepbiop

use deepbiop_core::default;
use pyo3::prelude::*;

// register default sub_module
pub fn register_default_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let sub_module_name = "default";
    let child_module = PyModule::new(parent_module.py(), sub_module_name)?;

    child_module.add("QUAL_OFFSET", default::QUAL_OFFSET)?;
    child_module.add("BASES", String::from_utf8_lossy(default::BASES))?;
    child_module.add("KMER_SIZE", default::KMER_SIZE)?;
    child_module.add("VECTORIZED_TARGET", default::VECTORIZED_TARGET)?;

    parent_module.add_submodule(&child_module)?;
    Ok(())
}
