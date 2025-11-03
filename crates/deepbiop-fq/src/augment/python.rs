//! Python bindings for augmentation operations.

use super::{Augmentation, Mutator, ReverseComplement, Sampler};
use pyo3::prelude::*;

/// Python wrapper for ReverseComplement.
#[pyclass(name = "ReverseComplement")]
pub struct PyReverseComplement {
    inner: ReverseComplement,
}

#[pymethods]
impl PyReverseComplement {
    /// Create a new reverse complement transformer for DNA.
    #[new]
    fn new() -> Self {
        Self {
            inner: ReverseComplement::new(),
        }
    }

    /// Create a new reverse complement transformer for RNA.
    #[staticmethod]
    fn for_rna() -> Self {
        Self {
            inner: ReverseComplement::for_rna(),
        }
    }

    /// Apply reverse complement transformation to a sequence.
    fn apply(&mut self, sequence: &[u8]) -> Vec<u8> {
        self.inner.apply(sequence)
    }

    fn __repr__(&self) -> String {
        "ReverseComplement()".to_string()
    }
}

/// Python wrapper for Mutator.
#[pyclass(name = "Mutator")]
pub struct PyMutator {
    inner: Mutator,
}

#[pymethods]
impl PyMutator {
    /// Create a new mutator with specified rate and optional seed.
    ///
    /// # Arguments
    ///
    /// * `mutation_rate` - Probability of mutating each base (0.0 to 1.0)
    /// * `seed` - Optional seed for reproducible mutations
    #[new]
    #[pyo3(signature = (mutation_rate, seed=None))]
    fn new(mutation_rate: f64, seed: Option<u64>) -> PyResult<Self> {
        Ok(Self {
            inner: Mutator::new(mutation_rate, seed),
        })
    }

    /// Create a mutator for RNA sequences.
    #[staticmethod]
    #[pyo3(signature = (mutation_rate, seed=None))]
    fn for_rna(mutation_rate: f64, seed: Option<u64>) -> PyResult<Self> {
        Ok(Self {
            inner: Mutator::for_rna(mutation_rate, seed),
        })
    }

    /// Apply mutation to a sequence.
    fn apply(&mut self, sequence: &[u8]) -> Vec<u8> {
        self.inner.apply(sequence)
    }

    fn __repr__(&self) -> String {
        "Mutator()".to_string()
    }
}

/// Python wrapper for Sampler.
#[pyclass(name = "Sampler")]
pub struct PySampler {
    inner: Sampler,
}

#[pymethods]
impl PySampler {
    /// Create a sampler that extracts subsequences from random positions.
    #[staticmethod]
    #[pyo3(signature = (length, seed=None))]
    fn random(length: usize, seed: Option<u64>) -> Self {
        Self {
            inner: Sampler::random(length, seed),
        }
    }

    /// Create a sampler that extracts subsequences from the start.
    #[staticmethod]
    fn from_start(length: usize) -> Self {
        Self {
            inner: Sampler::from_start(length),
        }
    }

    /// Create a sampler that extracts subsequences from the center.
    #[staticmethod]
    fn from_center(length: usize) -> Self {
        Self {
            inner: Sampler::from_center(length),
        }
    }

    /// Create a sampler that extracts subsequences from the end.
    #[staticmethod]
    fn from_end(length: usize) -> Self {
        Self {
            inner: Sampler::from_end(length),
        }
    }

    /// Apply sampling to a sequence.
    fn apply(&mut self, sequence: &[u8]) -> Vec<u8> {
        self.inner.apply(sequence)
    }

    fn __repr__(&self) -> String {
        format!("Sampler({:?})", self.inner)
    }
}

/// Register augmentation classes with Python module.
pub fn register_augmentation_classes(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyReverseComplement>()?;
    m.add_class::<PyMutator>()?;
    m.add_class::<PySampler>()?;
    Ok(())
}
