//! Python bindings for VCF processing

use crate::reader::VcfReader;
use crate::types::Variant;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use std::path::PathBuf;

#[gen_stub_pyclass]
#[pyclass(name = "VcfReader")]
/// Python wrapper for VCF file reader
pub struct PyVcfReader {
    inner: VcfReader,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyVcfReader {
    #[new]
    /// Open a VCF file for reading
    ///
    /// Args:
    ///     path: Path to the VCF file
    ///
    /// Returns:
    ///     VcfReader instance
    pub fn new(path: PathBuf) -> PyResult<Self> {
        let inner = VcfReader::open(&path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Read all variants from the file
    ///
    /// Returns:
    ///     List of Variant objects
    pub fn read_all(&mut self) -> PyResult<Vec<PyVariant>> {
        let variants = self
            .inner
            .read_all()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(variants.into_iter().map(PyVariant::from).collect())
    }

    /// Filter variants by minimum quality score
    ///
    /// Args:
    ///     min_quality: Minimum quality threshold
    ///
    /// Returns:
    ///     List of variants meeting quality criteria
    pub fn filter_by_quality(&mut self, min_quality: f32) -> PyResult<Vec<PyVariant>> {
        let variants = self
            .inner
            .filter_by_quality(min_quality)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(variants.into_iter().map(PyVariant::from).collect())
    }

    /// Filter variants that pass all filters
    ///
    /// Returns:
    ///     List of passing variants
    pub fn filter_passing(&mut self) -> PyResult<Vec<PyVariant>> {
        let variants = self
            .inner
            .filter_passing()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(variants.into_iter().map(PyVariant::from).collect())
    }
}

#[gen_stub_pyclass]
#[pyclass(name = "Variant")]
#[derive(Clone)]
/// Represents a genomic variant from a VCF file
pub struct PyVariant {
    #[pyo3(get)]
    pub chromosome: String,
    #[pyo3(get)]
    pub position: u64,
    #[pyo3(get)]
    pub id: Option<String>,
    #[pyo3(get)]
    pub reference_allele: String,
    #[pyo3(get)]
    pub alternate_alleles: Vec<String>,
    #[pyo3(get)]
    pub quality: Option<f32>,
    #[pyo3(get)]
    pub filter: Vec<String>,
}

impl From<Variant> for PyVariant {
    fn from(v: Variant) -> Self {
        Self {
            chromosome: v.chromosome,
            position: v.position,
            id: v.id,
            reference_allele: v.reference_allele,
            alternate_alleles: v.alternate_alleles,
            quality: v.quality,
            filter: v.filter,
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyVariant {
    /// Check if variant is a SNP (single nucleotide polymorphism)
    pub fn is_snp(&self) -> bool {
        self.reference_allele.len() == 1
            && self
                .alternate_alleles
                .iter()
                .all(|alt| alt.len() == 1 && alt != "*")
    }

    /// Check if variant is an indel (insertion/deletion)
    pub fn is_indel(&self) -> bool {
        let ref_len = self.reference_allele.len();
        self.alternate_alleles
            .iter()
            .any(|alt| alt.len() != ref_len || alt == "*")
    }

    /// Check if variant passes all filters
    pub fn passes_filter(&self) -> bool {
        self.filter.is_empty() || self.filter.contains(&"PASS".to_string())
    }

    fn __repr__(&self) -> String {
        format!(
            "Variant({}:{}, {} > {:?})",
            self.chromosome, self.position, self.reference_allele, self.alternate_alleles
        )
    }
}

/// Register VCF module with Python
pub fn register_vcf_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    parent_module.add_class::<PyVcfReader>()?;
    parent_module.add_class::<PyVariant>()?;
    Ok(())
}
