//! Python bindings for FASTQ filtering.

use super::{Deduplicator, Filter, LengthFilter, QualityFilter, Subsampler};
use noodles::fastq;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3_stub_gen::derive::*;

/// Python wrapper for LengthFilter.
#[cfg_attr(feature = "python", gen_stub_pyclass(module = "deepbiop.fq"))]
#[cfg_attr(feature = "python", pyclass(name = "LengthFilter"))]
pub struct PyLengthFilter {
    inner: LengthFilter,
}

#[cfg(feature = "python")]
#[gen_stub_pymethods]
#[cfg(feature = "python")]
#[pymethods]
impl PyLengthFilter {
    /// Create a new length filter.
    ///
    /// Args:
    ///     min_length: Minimum sequence length (inclusive), or None for no minimum
    ///     max_length: Maximum sequence length (inclusive), or None for no maximum
    ///
    /// Returns:
    ///     A new LengthFilter instance
    #[new]
    #[pyo3(signature = (min_length=None, max_length=None))]
    pub fn new(min_length: Option<usize>, max_length: Option<usize>) -> Self {
        Self {
            inner: LengthFilter::new(min_length, max_length),
        }
    }

    /// Create a filter that only accepts sequences with at least min bases.
    #[staticmethod]
    pub fn min_only(min: usize) -> Self {
        Self {
            inner: LengthFilter::min_only(min),
        }
    }

    /// Create a filter that only accepts sequences with at most max bases.
    #[staticmethod]
    pub fn max_only(max: usize) -> Self {
        Self {
            inner: LengthFilter::max_only(max),
        }
    }

    /// Create a filter that accepts sequences within a specific length range.
    #[staticmethod]
    pub fn range(min: usize, max: usize) -> Self {
        Self {
            inner: LengthFilter::range(min, max),
        }
    }

    /// Check if a record passes the filter.
    ///
    /// Args:
    ///     sequence: The sequence to check (bytes)
    ///
    /// Returns:
    ///     True if the sequence passes, False otherwise
    pub fn passes(&mut self, sequence: Vec<u8>) -> bool {
        let record = create_dummy_record(&sequence);
        self.inner.passes(&record)
    }

    /// Get the minimum length threshold.
    #[getter]
    pub fn min_length(&self) -> Option<usize> {
        self.inner.min_length()
    }

    /// Get the maximum length threshold.
    #[getter]
    pub fn max_length(&self) -> Option<usize> {
        self.inner.max_length()
    }

    /// String representation.
    pub fn __repr__(&self) -> String {
        format!(
            "LengthFilter(min_length={:?}, max_length={:?})",
            self.inner.min_length(),
            self.inner.max_length()
        )
    }
}

/// Python wrapper for QualityFilter.
#[cfg_attr(feature = "python", gen_stub_pyclass(module = "deepbiop.fq"))]
#[cfg_attr(feature = "python", pyclass(name = "QualityFilter"))]
pub struct PyQualityFilter {
    inner: QualityFilter,
}

#[cfg(feature = "python")]
#[gen_stub_pymethods]
#[cfg(feature = "python")]
#[pymethods]
impl PyQualityFilter {
    /// Create a new quality filter.
    ///
    /// Args:
    ///     min_mean_quality: Minimum mean quality score, or None for no minimum
    ///     min_base_quality: Minimum quality for any single base, or None for no minimum
    ///     quality_offset: Quality score encoding offset (typically 33 for Phred+33)
    ///
    /// Returns:
    ///     A new QualityFilter instance
    #[new]
    #[pyo3(signature = (min_mean_quality=None, min_base_quality=None, quality_offset=33))]
    pub fn new(
        min_mean_quality: Option<f64>,
        min_base_quality: Option<u8>,
        quality_offset: u8,
    ) -> Self {
        Self {
            inner: QualityFilter::new(min_mean_quality, min_base_quality, quality_offset),
        }
    }

    /// Create a filter based on mean quality only.
    #[staticmethod]
    #[pyo3(signature = (min_mean, quality_offset=33))]
    pub fn mean_quality(min_mean: f64, quality_offset: u8) -> Self {
        Self {
            inner: QualityFilter::mean_quality(min_mean, quality_offset),
        }
    }

    /// Create a filter based on minimum base quality only.
    #[staticmethod]
    #[pyo3(signature = (min_base, quality_offset=33))]
    pub fn base_quality(min_base: u8, quality_offset: u8) -> Self {
        Self {
            inner: QualityFilter::base_quality(min_base, quality_offset),
        }
    }

    /// Check if a record passes the filter.
    ///
    /// Args:
    ///     sequence: The sequence (bytes)
    ///     quality: The quality scores (bytes)
    ///
    /// Returns:
    ///     True if the record passes, False otherwise
    pub fn passes(&mut self, sequence: Vec<u8>, quality: Vec<u8>) -> PyResult<bool> {
        if sequence.len() != quality.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Sequence and quality must have the same length",
            ));
        }
        let record = fastq::Record::new(
            fastq::record::Definition::new("test", ""),
            sequence,
            quality,
        );
        Ok(self.inner.passes(&record))
    }

    /// Calculate mean quality score for a record.
    pub fn calculate_mean_quality(&self, quality: Vec<u8>) -> f64 {
        let record = fastq::Record::new(
            fastq::record::Definition::new("test", ""),
            vec![b'A'; quality.len()],
            quality,
        );
        self.inner.calculate_mean_quality(&record)
    }

    /// Get the quality offset.
    #[getter]
    pub fn quality_offset(&self) -> u8 {
        self.inner.quality_offset()
    }

    /// Get the minimum mean quality threshold.
    #[getter]
    pub fn min_mean_quality(&self) -> Option<f64> {
        self.inner.min_mean_quality()
    }

    /// Get the minimum base quality threshold.
    #[getter]
    pub fn min_base_quality(&self) -> Option<u8> {
        self.inner.min_base_quality()
    }

    /// String representation.
    pub fn __repr__(&self) -> String {
        format!(
            "QualityFilter(min_mean_quality={:?}, min_base_quality={:?}, offset={})",
            self.inner.min_mean_quality(),
            self.inner.min_base_quality(),
            self.inner.quality_offset()
        )
    }
}

/// Python wrapper for Deduplicator.
#[cfg_attr(feature = "python", gen_stub_pyclass(module = "deepbiop.fq"))]
#[cfg_attr(feature = "python", pyclass(name = "Deduplicator"))]
pub struct PyDeduplicator {
    inner: Deduplicator,
}

impl Default for PyDeduplicator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "python")]
#[gen_stub_pymethods]
#[cfg(feature = "python")]
#[pymethods]
impl PyDeduplicator {
    /// Create a new deduplicator.
    ///
    /// By default, keeps the first occurrence of each unique sequence.
    #[new]
    pub fn new() -> Self {
        Self {
            inner: Deduplicator::new(),
        }
    }

    /// Create a deduplicator that removes all duplicates (including first occurrence).
    #[staticmethod]
    pub fn remove_all_duplicates() -> Self {
        Self {
            inner: Deduplicator::remove_all_duplicates(),
        }
    }

    /// Check if a sequence has been seen before.
    pub fn is_duplicate(&self, sequence: Vec<u8>) -> bool {
        self.inner.is_duplicate(&sequence)
    }

    /// Check if a record passes the filter.
    ///
    /// Args:
    ///     sequence: The sequence to check (bytes)
    ///
    /// Returns:
    ///     True if the sequence passes (first occurrence), False if duplicate
    pub fn passes(&mut self, sequence: Vec<u8>) -> bool {
        let record = create_dummy_record(&sequence);
        self.inner.passes(&record)
    }

    /// Get the number of unique sequences seen so far.
    pub fn unique_count(&self) -> usize {
        self.inner.unique_count()
    }

    /// Clear all tracked sequences (reset the deduplicator).
    pub fn clear(&mut self) {
        self.inner.clear()
    }

    /// Get whether the filter keeps first occurrences.
    #[getter]
    pub fn keep_first(&self) -> bool {
        self.inner.keep_first()
    }

    /// String representation.
    pub fn __repr__(&self) -> String {
        format!(
            "Deduplicator(unique_count={}, keep_first={})",
            self.inner.unique_count(),
            self.inner.keep_first()
        )
    }
}

/// Python wrapper for Subsampler.
#[cfg_attr(feature = "python", gen_stub_pyclass(module = "deepbiop.fq"))]
#[cfg_attr(feature = "python", pyclass(name = "Subsampler"))]
pub struct PySubsampler {
    inner: Subsampler,
}

#[cfg(feature = "python")]
#[gen_stub_pymethods]
#[cfg(feature = "python")]
#[pymethods]
impl PySubsampler {
    /// Create a subsampler that keeps a random fraction of records.
    ///
    /// Args:
    ///     fraction: Fraction of records to keep (0.0 to 1.0)
    ///     seed: Optional seed for reproducible random sampling
    ///
    /// Returns:
    ///     A new Subsampler instance
    #[staticmethod]
    #[pyo3(signature = (fraction, seed=None))]
    pub fn random_fraction(fraction: f64, seed: Option<u64>) -> Self {
        Self {
            inner: Subsampler::random_fraction(fraction, seed),
        }
    }

    /// Create a subsampler that keeps every Nth record.
    #[staticmethod]
    pub fn every_nth(n: usize) -> Self {
        Self {
            inner: Subsampler::every_nth(n),
        }
    }

    /// Create a subsampler that keeps the first N records.
    #[staticmethod]
    pub fn first_n(n: usize) -> Self {
        Self {
            inner: Subsampler::first_n(n),
        }
    }

    /// Check if a record passes the filter.
    ///
    /// Args:
    ///     sequence: The sequence (bytes) - not used for subsampling but required for API consistency
    ///
    /// Returns:
    ///     True if this record should be kept based on the sampling strategy
    pub fn passes(&mut self, _sequence: Vec<u8>) -> bool {
        let record = create_dummy_record(b"A");
        self.inner.passes(&record)
    }

    /// Get the current record count.
    pub fn record_count(&self) -> usize {
        self.inner.record_count()
    }

    /// Reset the internal counter (useful for processing multiple files).
    pub fn reset(&mut self) {
        self.inner.reset()
    }

    /// String representation.
    pub fn __repr__(&self) -> String {
        format!("Subsampler(record_count={})", self.inner.record_count())
    }
}

/// Helper function to create a dummy FASTQ record for filtering.
#[cfg(feature = "python")]
fn create_dummy_record(sequence: &[u8]) -> fastq::Record {
    let quality = vec![b'I'; sequence.len()];
    fastq::Record::new(
        fastq::record::Definition::new("test", ""),
        sequence.to_vec(),
        quality,
    )
}
