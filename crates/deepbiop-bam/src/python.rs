//! Python bindings for BAM processing
//!
//! Note: Uses std::collections::HashMap instead of ahash::HashMap
//! because PyO3 requires std HashMap for Python interoperability.

// Allow std::HashMap for PyO3 compatibility
#![allow(clippy::disallowed_types)]

use std::collections::HashMap as StdHashMap;
use std::path::PathBuf;

use ahash::HashMap;
use anyhow::Result;
use noodles::sam::record::Cigar;
use pyo3::prelude::*;

use crate::chimeric;
use crate::cigar::calc_softclips;
use crate::features::AlignmentFeatures;
use crate::reader::BamReader;

use pyo3_stub_gen::derive::*;

/// Python wrapper for AlignmentFeatures
#[gen_stub_pyclass]
#[pyclass(name = "AlignmentFeatures", module = "deepbiop.bam")]
#[derive(Clone)]
pub struct PyAlignmentFeatures {
    inner: AlignmentFeatures,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyAlignmentFeatures {
    #[getter]
    fn mapping_quality(&self) -> u8 {
        self.inner.mapping_quality
    }

    #[getter]
    fn is_mapped(&self) -> bool {
        self.inner.is_mapped
    }

    #[getter]
    fn is_paired(&self) -> bool {
        self.inner.is_paired
    }

    #[getter]
    fn is_supplementary(&self) -> bool {
        self.inner.is_supplementary
    }

    #[getter]
    fn is_secondary(&self) -> bool {
        self.inner.is_secondary
    }

    #[getter]
    fn is_mate_mapped(&self) -> Option<bool> {
        self.inner.is_mate_mapped
    }

    #[getter]
    fn template_length(&self) -> i32 {
        self.inner.template_length
    }

    #[getter]
    fn aligned_length(&self) -> usize {
        self.inner.aligned_length
    }

    #[getter]
    fn num_matches(&self) -> usize {
        self.inner.num_matches
    }

    #[getter]
    fn num_insertions(&self) -> usize {
        self.inner.num_insertions
    }

    #[getter]
    fn num_deletions(&self) -> usize {
        self.inner.num_deletions
    }

    #[getter]
    fn num_soft_clips(&self) -> usize {
        self.inner.num_soft_clips
    }

    #[getter]
    fn num_hard_clips(&self) -> usize {
        self.inner.num_hard_clips
    }

    #[getter]
    fn edit_distance(&self) -> Option<u32> {
        self.inner.edit_distance
    }

    #[getter]
    fn tags(&self) -> StdHashMap<String, String> {
        self.inner
            .tags
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }

    fn identity(&self) -> f32 {
        self.inner.identity()
    }

    fn indel_rate(&self) -> f32 {
        self.inner.indel_rate()
    }

    #[pyo3(signature = (min_quality))]
    fn is_high_quality(&self, min_quality: u8) -> bool {
        self.inner.is_high_quality(min_quality)
    }

    #[pyo3(signature = (max_insert_size))]
    fn is_proper_pair(&self, max_insert_size: i32) -> bool {
        self.inner.is_proper_pair(max_insert_size)
    }

    fn __repr__(&self) -> String {
        format!(
            "AlignmentFeatures(mapq={}, aligned_length={}, identity={:.3})",
            self.inner.mapping_quality,
            self.inner.aligned_length,
            self.inner.identity()
        )
    }
}

impl From<AlignmentFeatures> for PyAlignmentFeatures {
    fn from(inner: AlignmentFeatures) -> Self {
        Self { inner }
    }
}

/// Python wrapper for BamReader
#[gen_stub_pyclass]
#[pyclass(name = "BamReader", module = "deepbiop.bam")]
pub struct PyBamReader {
    inner: BamReader,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyBamReader {
    #[new]
    #[pyo3(signature = (path, threads=None))]
    fn new(path: PathBuf, threads: Option<usize>) -> Result<Self> {
        let reader = BamReader::open(path, threads)?;
        Ok(Self { inner: reader })
    }

    #[pyo3(signature = (min_quality))]
    fn filter_by_mapping_quality(&mut self, min_quality: u8) -> Result<usize> {
        let records = self.inner.filter_by_mapping_quality(min_quality)?;
        Ok(records.len())
    }

    fn extract_read_pairs(&mut self) -> Result<usize> {
        let records = self.inner.extract_read_pairs()?;
        Ok(records.len())
    }

    fn extract_features(&mut self) -> Result<Vec<PyAlignmentFeatures>> {
        let features = self.inner.extract_features()?;
        Ok(features.into_iter().map(|f| f.into()).collect())
    }

    fn count_chimeric(&mut self) -> Result<usize> {
        self.inner.count_chimeric()
    }

    fn __repr__(&self) -> String {
        "BamReader(...)".to_string()
    }
}

/// Calculate the number of chimeric reads in a BAM file.
#[gen_stub_pyfunction(module = "deepbiop.bam")]
#[pyfunction]
#[pyo3(signature = (bam, threads=None))]
fn count_chimeric_reads_for_path(bam: PathBuf, threads: Option<usize>) -> Result<usize> {
    chimeric::count_chimeric_reads_for_path(bam, threads)
}

/// Calculate the number of chimeric reads in multiple BAM files.
#[gen_stub_pyfunction(module = "deepbiop.bam")]
#[pyfunction]
#[pyo3(signature = (bams, threads=None))]
fn count_chimeric_reads_for_paths(
    bams: Vec<PathBuf>,
    threads: Option<usize>,
) -> Result<HashMap<PathBuf, usize>> {
    Ok(chimeric::count_chimeric_reads_for_paths(&bams, threads))
}

/// Calculate left and right soft clips from a cigar string.
#[gen_stub_pyfunction(module = "deepbiop.bam")]
#[pyfunction]
fn left_right_soft_clip(cigar_string: &str) -> Result<(usize, usize)> {
    let cigar = Cigar::new(cigar_string.as_bytes());
    calc_softclips(&cigar)
}

// register bam sub module
pub fn register_bam_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let sub_module_name = "bam";
    let child_module = PyModule::new(parent_module.py(), sub_module_name)?;

    // Add classes
    child_module.add_class::<PyAlignmentFeatures>()?;
    child_module.add_class::<PyBamReader>()?;

    // Add functions
    child_module.add_function(wrap_pyfunction!(left_right_soft_clip, &child_module)?)?;
    child_module.add_function(wrap_pyfunction!(
        count_chimeric_reads_for_path,
        &child_module
    )?)?;
    child_module.add_function(wrap_pyfunction!(
        count_chimeric_reads_for_paths,
        &child_module
    )?)?;

    parent_module.add_submodule(&child_module)?;
    Ok(())
}
