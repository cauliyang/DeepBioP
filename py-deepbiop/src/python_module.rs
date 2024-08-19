//! Python module for deepbiop

use pyo3::prelude::*;

// register default sub_module
pub fn register_default_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let sub_module_name = "default";
    let child_module = PyModule::new_bound(parent_module.py(), sub_module_name)?;

    child_module.add("QUAL_OFFSET", deepbiop_fq::default::QUAL_OFFSET)?;
    child_module.add(
        "BASES",
        String::from_utf8_lossy(deepbiop_fq::default::BASES),
    )?;
    child_module.add("KMER_SIZE", deepbiop_fq::default::KMER_SIZE)?;
    child_module.add("VECTORIZED_TARGET", deepbiop_fq::default::VECTORIZED_TARGET)?;

    parent_module.add_submodule(&child_module)?;
    Ok(())
}
