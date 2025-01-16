mod python_module;

use pyo3::prelude::*;

use deepbiop_bam::python::register_bam_module;
use deepbiop_core::python::register_core_module;
use deepbiop_fa::python::register_fa_module;
use deepbiop_fq::python::register_fq_module;
use deepbiop_utils::python::register_utils_module;

use pyo3_stub_gen::define_stub_info_gatherer;

#[pymodule]
fn deepbiop(m: &Bound<'_, PyModule>) -> PyResult<()> {
    pyo3_log::init();

    python_module::register_default_module(m)?;
    register_fq_module(m)?;
    register_bam_module(m)?;
    register_utils_module(m)?;
    register_fa_module(m)?;
    register_core_module(m)?;

    Ok(())
}

// Define a function to gather stub information.
define_stub_info_gatherer!(stub_info);
