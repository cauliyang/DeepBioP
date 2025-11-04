//! Dataset class for PyTorch-style data loading.
//!
//! This module provides the Dataset class that wraps DeepBioP file readers
//! and presents a PyTorch-compatible interface for loading FASTQ/FASTA files.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::path::Path;

use deepbiop_fq::dataset::FastqDataset as InnerFastqDataset;

/// PyTorch-compatible Dataset for biological sequence files.
///
/// Wraps existing FastqDataset to provide PyTorch Dataset interface:
/// - __len__(): Returns total number of sequences
/// - __getitem__(): Returns individual samples as dicts
/// - __iter__(): Returns iterator over samples
#[pyclass(name = "Dataset", module = "deepbiop.pytorch", sequence)]
pub struct Dataset {
    /// Inner FASTQ dataset (wraps existing implementation)
    inner: InnerFastqDataset,
    /// Total number of records
    num_records: usize,
    /// File path for repr
    file_path: String,
}

#[pymethods]
impl Dataset {
    /// Create a new Dataset from FASTQ/FASTA file(s).
    ///
    /// Args:
    ///     file_paths: Path to FASTQ/FASTA file (or list of paths)
    ///     sequence_type: Type of sequences ("dna", "rna", "protein") - currently unused
    ///     transform: Optional transformation pipeline - currently unused
    ///     cache_dir: Directory for caching - currently unused
    ///     lazy: Load sequences on-demand (default: True) - currently unused
    ///
    /// Returns:
    ///     Dataset instance
    ///
    /// Raises:
    ///     FileNotFoundError: If file doesn't exist
    ///     ValueError: If file is invalid format
    #[new]
    #[pyo3(signature = (file_paths, *, _sequence_type="dna", _transform=None, _cache_dir=None, _lazy=true))]
    #[allow(unused_variables)]
    fn new(
        file_paths: &Bound<'_, PyAny>,
        _sequence_type: &str,
        _transform: Option<Py<PyAny>>,
        _cache_dir: Option<String>,
        _lazy: bool,
    ) -> PyResult<Self> {
        // For now, only support single file path (string)
        let file_path = if let Ok(path_str) = file_paths.extract::<String>() {
            path_str
        } else {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Currently only single file path (string) is supported",
            ));
        };

        // Check file exists
        if !Path::new(&file_path).exists() {
            return Err(pyo3::exceptions::PyFileNotFoundError::new_err(format!(
                "FASTQ/FASTA file '{}' not found. Check path and permissions.",
                file_path
            )));
        }

        // Create inner FastqDataset with chunk_size=1 for individual record access
        let inner = InnerFastqDataset::new(file_path.clone(), 1)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

        // Get total number of records
        let num_records = inner.records_count();

        // TODO: Handle transform, cache_dir, lazy parameters in future tasks

        Ok(Dataset {
            inner,
            num_records,
            file_path,
        })
    }

    /// Returns total number of sequences in dataset.
    fn __len__(&self) -> PyResult<usize> {
        Ok(self.num_records)
    }

    /// Get sample at index idx.
    ///
    /// Args:
    ///     idx: Sample index (0 to len(dataset)-1)
    ///
    /// Returns:
    ///     Sample dict with 'sequence' and 'quality' keys
    ///
    /// Raises:
    ///     IndexError: If idx out of range
    fn __getitem__(&self, idx: usize, py: Python) -> PyResult<Py<PyDict>> {
        if idx >= self.num_records {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "Index {} out of range (dataset has {} sequences)",
                idx, self.num_records
            )));
        }

        // Get single record from inner dataset
        let records = self
            .inner
            .get_records(idx, idx + 1)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

        if records.is_empty() {
            return Err(pyo3::exceptions::PyIOError::new_err(format!(
                "Failed to read record at index {}",
                idx
            )));
        }

        let record = &records[0];

        // Create Sample dict
        let sample = PyDict::new(py);
        sample.set_item("sequence", record.sequence.as_bytes())?;
        sample.set_item("quality", record.quality.as_bytes())?;

        Ok(sample.into())
    }

    /// Iterate over all samples in dataset.
    ///
    /// Returns:
    ///     Iterator over Sample dicts
    fn __iter__(slf: PyRef<'_, Self>) -> PyResult<DatasetIterator> {
        Ok(DatasetIterator {
            dataset: slf.into(),
            current_idx: 0,
        })
    }

    /// Human-readable representation.
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "Dataset(num_samples={}, sequence_type='dna', transform=None)",
            self.num_records
        ))
    }

    /// Get dataset statistics.
    ///
    /// Returns:
    ///     Dict with keys:
    ///         - 'num_samples': int - Total number of sequences
    ///         - 'length_stats': dict - Statistics with 'min', 'max', 'mean', 'median' sequence lengths
    ///         - 'memory_footprint': int - Estimated memory usage in bytes (sequence + quality)
    ///
    /// Examples:
    ///     >>> dataset.summary()
    ///     {'num_samples': 1000, 'length_stats': {'min': 100, 'max': 150, 'mean': 125.5, 'median': 125.0}, 'memory_footprint': 250000}
    fn summary(&self, py: Python) -> PyResult<Py<PyDict>> {
        // Collect all sequence lengths
        let mut lengths = Vec::with_capacity(self.num_records);

        for idx in 0..self.num_records {
            let records = self
                .inner
                .get_records(idx, idx + 1)
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

            if !records.is_empty() {
                lengths.push(records[0].sequence.len());
            }
        }

        // Calculate statistics
        if lengths.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Dataset is empty, cannot generate summary",
            ));
        }

        let min_len = *lengths.iter().min().unwrap();
        let max_len = *lengths.iter().max().unwrap();
        let sum: usize = lengths.iter().sum();
        let mean_len = sum as f64 / lengths.len() as f64;

        // Calculate median
        let mut sorted_lengths = lengths.clone();
        sorted_lengths.sort_unstable();
        let median_len = if sorted_lengths.len() % 2 == 0 {
            let mid = sorted_lengths.len() / 2;
            (sorted_lengths[mid - 1] + sorted_lengths[mid]) as f64 / 2.0
        } else {
            sorted_lengths[sorted_lengths.len() / 2] as f64
        };

        // Estimate memory footprint (sequence + quality bytes)
        let memory_footprint: usize = lengths.iter().map(|&len| len * 2).sum();

        // Build summary dict
        let summary = PyDict::new(py);
        summary.set_item("num_samples", self.num_records)?;

        let length_stats = PyDict::new(py);
        length_stats.set_item("min", min_len)?;
        length_stats.set_item("max", max_len)?;
        length_stats.set_item("mean", mean_len)?;
        length_stats.set_item("median", median_len)?;
        summary.set_item("length_stats", length_stats)?;

        summary.set_item("memory_footprint", memory_footprint)?;

        Ok(summary.into())
    }

    /// Validate dataset quality and integrity.
    ///
    /// Performs validation checks on a sample of records (first 10):
    /// - Empty sequences
    /// - Invalid DNA/RNA bases
    /// - Quality string length mismatches
    ///
    /// Returns:
    ///     Dict with keys:
    ///         - 'is_valid': bool - True if no errors found
    ///         - 'warnings': list[str] - Non-critical issues
    ///         - 'errors': list[str] - Critical issues that prevent use
    ///
    /// Examples:
    ///     >>> dataset.validate()
    ///     {'is_valid': True, 'warnings': [], 'errors': []}
    fn validate(&self, py: Python) -> PyResult<Py<PyDict>> {
        let mut warnings: Vec<String> = Vec::new();
        let mut errors: Vec<String> = Vec::new();

        // Check if dataset is empty
        if self.num_records == 0 {
            errors.push("Dataset is empty".to_string());
        }

        // Validate a sample of records (check first 10 or all if fewer)
        let sample_size = std::cmp::min(10, self.num_records);

        for idx in 0..sample_size {
            let records = self
                .inner
                .get_records(idx, idx + 1)
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

            if records.is_empty() {
                errors.push(format!("Failed to read record at index {}", idx));
                continue;
            }

            let record = &records[0];

            // Check for empty sequence
            if record.sequence.is_empty() {
                errors.push(format!("Empty sequence at index {}", idx));
            }

            // Check if quality length matches sequence length
            if record.sequence.len() != record.quality.len() {
                warnings.push(format!(
                    "Quality length ({}) doesn't match sequence length ({}) at index {}",
                    record.quality.len(),
                    record.sequence.len(),
                    idx
                ));
            }

            // Check for valid DNA/RNA bases (basic validation)
            let valid_bases = b"ACGTNacgtnUu-";
            for &base in record.sequence.as_bytes() {
                if !valid_bases.contains(&base) {
                    warnings.push(format!(
                        "Unexpected base '{}' at index {}",
                        char::from(base),
                        idx
                    ));
                    break;
                }
            }
        }

        // If we sampled less than the full dataset, add a note
        if sample_size < self.num_records {
            warnings.push(format!(
                "Validated {} of {} sequences (sample validation)",
                sample_size, self.num_records
            ));
        }

        // Build validation result dict
        let result = PyDict::new(py);
        result.set_item("is_valid", errors.is_empty())?;
        result.set_item("warnings", warnings)?;
        result.set_item("errors", errors)?;

        Ok(result.into())
    }
}

// Public Rust API for internal use by DataLoader
impl Dataset {
    /// Get dataset length (for Rust internal use).
    pub fn len(&self) -> usize {
        self.num_records
    }

    /// Get sample at index (for Rust internal use).
    pub fn get_item(&self, idx: usize, py: Python) -> PyResult<Py<PyDict>> {
        self.__getitem__(idx, py)
    }
}

/// Iterator for Dataset.
///
/// Separate iterator class that maintains iteration state.
#[pyclass(name = "DatasetIterator", module = "deepbiop.pytorch")]
pub struct DatasetIterator {
    dataset: Py<Dataset>,
    current_idx: usize,
}

#[pymethods]
impl DatasetIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python) -> PyResult<Option<Py<PyDict>>> {
        let dataset = self.dataset.borrow(py);

        if self.current_idx >= dataset.num_records {
            return Ok(None);
        }

        let sample = dataset.__getitem__(self.current_idx, py)?;
        self.current_idx += 1;

        Ok(Some(sample))
    }
}
