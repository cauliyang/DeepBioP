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
use numpy::convert::ToPyArray;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::chimeric;
use crate::cigar::calc_softclips;
use crate::features::AlignmentFeatures;
use crate::reader::BamReader;

use deepbiop_core::dataset::IterableDataset;
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

/// Streaming BAM dataset for efficient iteration over large files.
///
/// This dataset provides memory-efficient streaming iteration over BAM files,
/// reading alignment records one at a time without loading the entire file into memory.
/// Yields records as dictionaries with NumPy arrays for zero-copy efficiency.
/// Supports multithreaded bgzf decompression for improved performance.
///
/// # Examples
///
/// ```python
/// from deepbiop.bam import BamStreamDataset
///
/// dataset = BamStreamDataset("alignments.bam", threads=4)
/// for record in dataset:
///     seq = record['sequence']  # NumPy array
///     qual = record['quality']   # NumPy array
///     print(f"ID: {record['id']}, Length: {len(seq)}")
/// ```
#[gen_stub_pyclass]
#[pyclass(name = "BamStreamDataset", module = "deepbiop.bam")]
pub struct PyBamStreamDataset {
    file_path: String,
    threads: Option<usize>,
    size_hint: Option<usize>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyBamStreamDataset {
    /// Create a new streaming BAM dataset.
    ///
    /// # Arguments
    ///
    /// * `file_path` - Path to BAM file
    /// * `threads` - Optional number of threads for bgzf decompression (None = use all available)
    ///
    /// # Returns
    ///
    /// BamStreamDataset instance
    ///
    /// # Raises
    ///
    /// * `FileNotFoundError` - If file doesn't exist
    /// * `IOError` - If file cannot be opened
    #[new]
    #[pyo3(signature = (file_path, threads=None))]
    fn new(file_path: String, threads: Option<usize>) -> PyResult<Self> {
        // Create temporary dataset to validate file
        let dataset = crate::dataset::BamDataset::new(file_path.clone(), threads)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

        let size_hint = dataset.records_count();

        Ok(Self {
            file_path,
            threads,
            size_hint,
        })
    }

    /// Return iterator over dataset records.
    ///
    /// Each record is a dictionary with:
    /// - 'id': str - Read name/identifier
    /// - 'sequence': np.ndarray (uint8) - Nucleotide sequence bytes
    /// - 'quality': np.ndarray (uint8) - Quality score bytes
    /// - 'description': Optional[str] - Additional description (usually None for BAM)
    fn __iter__(slf: PyRef<'_, Self>) -> PyResult<Py<PyBamStreamIterator>> {
        let iter = PyBamStreamIterator {
            file_path: slf.file_path.clone(),
            threads: slf.threads,
            current_idx: 0,
        };
        Py::new(slf.py(), iter)
    }

    /// Return the number of records in the dataset.
    ///
    /// This is computed once at dataset creation by reading through the file.
    /// Required for PyTorch DataLoader compatibility.
    fn __len__(&self) -> PyResult<usize> {
        Ok(self.size_hint.unwrap_or(0))
    }

    /// Get record by index (map-style dataset access).
    ///
    /// # Arguments
    ///
    /// * `index` - Index of record to retrieve
    ///
    /// # Returns
    ///
    /// Dict with 'id', 'sequence', 'quality', 'description'
    ///
    /// # Performance Warning
    ///
    /// **This implementation has O(n) complexity and reopens the BAM file for every access.**
    ///
    /// - Accessing index 1000 requires reading and discarding 1000 records
    /// - Each call reopens the file (costly I/O operation)
    /// - No caching or state preservation between calls
    ///
    /// This is acceptable for PyTorch DataLoader which uses sequential access patterns,
    /// but will be extremely inefficient for random access patterns. If you need random
    /// access to multiple indices, consider loading all records into memory first or
    /// using sequential iteration instead.
    ///
    /// For batch access, use the iterator interface which provides true streaming.
    fn __getitem__(&self, index: usize, py: Python) -> PyResult<Py<PyDict>> {
        use numpy::ToPyArray;

        // Create dataset and iterate to index
        let dataset = crate::dataset::BamDataset::new(self.file_path.clone(), self.threads)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

        let mut iter = dataset.iter();

        // Skip to index
        for _ in 0..index {
            if iter.next().is_none() {
                return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                    "Index {} out of range for dataset with {} records",
                    index,
                    self.size_hint.unwrap_or(0)
                )));
            }
        }

        // Get record at index
        match iter.next() {
            None => Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "Index {} out of range",
                index
            ))),
            Some(Ok(record)) => {
                let dict = PyDict::new(py);
                dict.set_item("id", record.id)?;

                // Convert sequence and quality to NumPy arrays
                let seq_array = record.sequence.to_pyarray(py);
                dict.set_item("sequence", seq_array)?;

                if let Some(quality) = record.quality_scores {
                    let qual_array = quality.to_pyarray(py);
                    dict.set_item("quality", qual_array)?;
                } else {
                    dict.set_item("quality", py.None())?;
                }

                dict.set_item("description", record.description)?;

                Ok(dict.into())
            }
            Some(Err(e)) => Err(pyo3::exceptions::PyIOError::new_err(format!(
                "Failed to read BAM record: {}",
                e
            ))),
        }
    }

    /// Human-readable representation.
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "BamStreamDataset(file='{}', threads={}, records={})",
            self.file_path,
            self.threads.map_or("auto".to_string(), |n| n.to_string()),
            self.size_hint
                .map_or("unknown".to_string(), |n| n.to_string())
        ))
    }

    /// Pickling support for multiprocessing (DataLoader with num_workers > 0).
    fn __getstate__(&self, py: Python) -> PyResult<Py<PyDict>> {
        let state = PyDict::new(py);
        state.set_item("file_path", &self.file_path)?;
        state.set_item("threads", self.threads)?;
        state.set_item("size_hint", self.size_hint)?;
        Ok(state.into())
    }

    /// Unpickling support for multiprocessing.
    fn __setstate__(&mut self, state: &Bound<'_, PyDict>) -> PyResult<()> {
        self.file_path = state.get_item("file_path")?.unwrap().extract()?;
        self.threads = state.get_item("threads")?.unwrap().extract()?;
        self.size_hint = state.get_item("size_hint")?.unwrap().extract()?;
        Ok(())
    }
}

/// Iterator for streaming BAM dataset.
#[gen_stub_pyclass]
#[pyclass(name = "BamStreamIterator", module = "deepbiop.bam")]
pub struct PyBamStreamIterator {
    file_path: String,
    threads: Option<usize>,
    current_idx: usize,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyBamStreamIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python) -> PyResult<Option<Py<PyDict>>> {
        // Create a new dataset and iterator for each record
        let dataset = crate::dataset::BamDataset::new(self.file_path.clone(), self.threads)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

        // Skip to current index and get next record
        let mut iter = dataset.iter();
        for _ in 0..self.current_idx {
            if iter.next().is_none() {
                return Ok(None);
            }
        }

        match iter.next() {
            None => Ok(None),
            Some(Ok(record)) => {
                self.current_idx += 1;

                // Create dict with NumPy arrays for zero-copy
                let dict = PyDict::new(py);
                dict.set_item("id", record.id)?;

                // Convert sequence and quality to NumPy arrays (zero-copy from Rust Vec)
                let seq_array = record.sequence.to_pyarray(py);
                dict.set_item("sequence", seq_array)?;

                if let Some(quality) = record.quality_scores {
                    let qual_array = quality.to_pyarray(py);
                    dict.set_item("quality", qual_array)?;
                } else {
                    dict.set_item("quality", py.None())?;
                }

                dict.set_item("description", record.description)?;

                Ok(Some(dict.into()))
            }
            Some(Err(e)) => Err(pyo3::exceptions::PyIOError::new_err(format!(
                "Failed to read BAM record: {}",
                e
            ))),
        }
    }
}

// register bam sub module
pub fn register_bam_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let sub_module_name = "bam";
    let child_module = PyModule::new(parent_module.py(), sub_module_name)?;

    // Add classes
    child_module.add_class::<PyAlignmentFeatures>()?;
    child_module.add_class::<PyBamReader>()?;

    // Add streaming dataset classes
    child_module.add_class::<PyBamStreamDataset>()?;
    child_module.add_class::<PyBamStreamIterator>()?;

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
