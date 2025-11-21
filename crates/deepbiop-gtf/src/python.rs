//! Python bindings for GTF processing
//!
//! Note: Uses std::collections::HashMap instead of ahash::HashMap
//! because PyO3 requires std HashMap for Python interoperability.

// Allow std::HashMap for PyO3 compatibility
#![allow(clippy::disallowed_types)]

use crate::reader::GtfReader;
use crate::types::{GenomicFeature, Strand};
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use std::collections::HashMap;
use std::path::PathBuf;

#[gen_stub_pyclass]
#[pyclass(name = "GtfReader", module = "deepbiop.gtf")]
/// Python wrapper for GTF file reader
pub struct PyGtfReader {
    inner: GtfReader,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyGtfReader {
    #[new]
    /// Open a GTF file for reading
    ///
    /// Args:
    ///     path: Path to the GTF file
    ///
    /// Returns:
    ///     GtfReader instance
    pub fn new(path: PathBuf) -> PyResult<Self> {
        let inner = GtfReader::open(&path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Read all features from the file
    ///
    /// Returns:
    ///     List of GenomicFeature objects
    pub fn read_all(&mut self) -> PyResult<Vec<PyGenomicFeature>> {
        let features = self
            .inner
            .read_all()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(features.into_iter().map(PyGenomicFeature::from).collect())
    }

    /// Build an index of features by gene ID
    ///
    /// Returns:
    ///     Dictionary mapping gene IDs to lists of features
    pub fn build_gene_index(&mut self) -> PyResult<HashMap<String, Vec<PyGenomicFeature>>> {
        let index = self
            .inner
            .build_gene_index()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(index
            .into_iter()
            .map(|(k, v)| (k, v.into_iter().map(PyGenomicFeature::from).collect()))
            .collect())
    }

    /// Filter features by type (e.g., "gene", "exon", "CDS")
    ///
    /// Args:
    ///     feature_type: Feature type to filter for
    ///
    /// Returns:
    ///     List of features of the specified type
    pub fn filter_by_type(&mut self, feature_type: String) -> PyResult<Vec<PyGenomicFeature>> {
        let features = self
            .inner
            .filter_by_type(&feature_type)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(features.into_iter().map(PyGenomicFeature::from).collect())
    }
}

#[gen_stub_pyclass]
#[pyclass(name = "GenomicFeature", module = "deepbiop.gtf")]
#[derive(Clone)]
/// Represents a genomic feature from a GTF file
pub struct PyGenomicFeature {
    #[pyo3(get)]
    pub seqname: String,
    #[pyo3(get)]
    pub source: String,
    #[pyo3(get)]
    pub feature_type: String,
    #[pyo3(get)]
    pub start: u64,
    #[pyo3(get)]
    pub end: u64,
    #[pyo3(get)]
    pub score: Option<f32>,
    #[pyo3(get)]
    pub strand: String,
    #[pyo3(get)]
    pub frame: Option<u8>,
    #[pyo3(get)]
    pub attributes: HashMap<String, String>,
}

impl From<GenomicFeature> for PyGenomicFeature {
    fn from(f: GenomicFeature) -> Self {
        let strand_str = match f.strand {
            Strand::Forward => "+".to_string(),
            Strand::Reverse => "-".to_string(),
            Strand::Unstranded => ".".to_string(),
        };

        // Convert ahash::HashMap to std::collections::HashMap
        let attributes: HashMap<String, String> = f.attributes.into_iter().collect();

        Self {
            seqname: f.seqname,
            source: f.source,
            feature_type: f.feature_type,
            start: f.start,
            end: f.end,
            score: f.score,
            strand: strand_str,
            frame: f.frame,
            attributes,
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyGenomicFeature {
    /// Get gene ID from attributes
    ///
    /// Returns:
    ///     Gene ID if present, None otherwise
    pub fn gene_id(&self) -> Option<String> {
        self.attributes.get("gene_id").cloned()
    }

    /// Get transcript ID from attributes
    ///
    /// Returns:
    ///     Transcript ID if present, None otherwise
    pub fn transcript_id(&self) -> Option<String> {
        self.attributes.get("transcript_id").cloned()
    }

    /// Get gene name from attributes
    ///
    /// Returns:
    ///     Gene name if present, None otherwise
    pub fn gene_name(&self) -> Option<String> {
        self.attributes.get("gene_name").cloned()
    }

    /// Get the length of the feature
    ///
    /// Returns:
    ///     Length in base pairs
    pub fn length(&self) -> u64 {
        if self.end >= self.start {
            self.end - self.start + 1
        } else {
            0
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "GenomicFeature({}: {}:{}-{} [{}])",
            self.feature_type, self.seqname, self.start, self.end, self.strand
        )
    }
}

/// Register GTF module with Python
pub fn register_gtf_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let gtf_module = PyModule::new(parent_module.py(), "gtf")?;

    // Module docstring
    gtf_module.add(
        "__doc__",
        "GTF (Gene Transfer Format) annotation file processing.\n\n\
         Provides GtfReader and GenomicFeature classes for reading and analyzing genomic annotations.",
    )?;

    // Register classes
    gtf_module.add_class::<PyGtfReader>()?;
    gtf_module.add_class::<PyGenomicFeature>()?;

    // Add as submodule
    parent_module.add_submodule(&gtf_module)?;

    Ok(())
}
