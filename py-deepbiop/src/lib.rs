mod python_module;

use pyo3::prelude::*;

use deepbiop_bam::python::register_bam_module;
use deepbiop_core::python::register_core_module;
use deepbiop_fa::python::register_fa_module;
use deepbiop_fq::python::register_fq_module;
use deepbiop_gtf::python::register_gtf_module;
use deepbiop_utils::python::register_utils_module;
use deepbiop_vcf::python::register_vcf_module;
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
    register_vcf_module(m)?;
    register_gtf_module(m)?;

    // Re-export commonly used classes at top level for convenience
    // Augmentation classes
    let fq_module = m.getattr("fq")?;
    m.add("ReverseComplement", fq_module.getattr("ReverseComplement")?)?;
    m.add("Mutator", fq_module.getattr("Mutator")?)?;
    m.add("Sampler", fq_module.getattr("Sampler")?)?;
    m.add("QualityModel", fq_module.getattr("QualityModel")?)?;
    m.add("QualitySimulator", fq_module.getattr("QualitySimulator")?)?;

    // Encoding classes
    m.add("OneHotEncoder", fq_module.getattr("OneHotEncoder")?)?;
    m.add("IntegerEncoder", fq_module.getattr("IntegerEncoder")?)?;

    let core_module = m.getattr("core")?;
    m.add("KmerEncoder", core_module.getattr("KmerEncoder")?)?;

    Ok(())
}

// Define a function to gather stub information.
define_stub_info_gatherer!(stub_info);
