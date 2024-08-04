mod python_module;

use pyo3::prelude::*;

use deepbiop_bam::python::register_bam_module;
use deepbiop_fq::python::register_fq_module;
use deepbiop_utils::python::register_utils_module;

#[pyfunction]
fn add(a: usize, b: usize) -> usize {
    a + b
}

#[pymodule]
fn deepbiop(m: &Bound<'_, PyModule>) -> PyResult<()> {
    pyo3_log::init();

    m.add_function(wrap_pyfunction!(add, m)?)?;

    python_module::register_default_module(m)?;
    register_fq_module(m)?;
    register_bam_module(m)?;
    register_utils_module(m)?;

    Ok(())
}
