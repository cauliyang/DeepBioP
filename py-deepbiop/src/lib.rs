mod python_module;

use pyo3::prelude::*;

#[pyfunction]
fn add(a: usize, b: usize) -> usize {
    a + b
}

#[pymodule]
fn deepbiop(m: &Bound<'_, PyModule>) -> PyResult<()> {
    pyo3_log::init();
    python_module::register_default_module(m)?;
    python_module::register_fq_module(m)?;
    python_module::register_bam_module(m)?;

    m.add_function(wrap_pyfunction!(add, m)?)?;

    Ok(())
}
