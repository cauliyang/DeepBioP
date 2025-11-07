//! Python bindings for augmentation operations.

use super::{
    quality::QualityModel, quality::QualitySimulator, Augmentation, Mutator, ReverseComplement,
    Sampler,
};
#[cfg(feature = "python")]
use pyo3::prelude::*;

/// Python wrapper for ReverseComplement.
#[cfg_attr(feature = "python", pyclass(name = "ReverseComplement"))]
pub struct PyReverseComplement {
    inner: ReverseComplement,
}

#[cfg(feature = "python")]
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

    /// Apply reverse complement to multiple sequences.
    fn apply_batch(&mut self, py: Python, sequences: Vec<Vec<u8>>) -> Vec<Vec<u8>> {
        // Release GIL for parallel processing with Rayon
        py.detach(|| self.inner.apply_batch(&sequences))
    }

    fn __repr__(&self) -> String {
        "ReverseComplement()".to_string()
    }
}

/// Python wrapper for Mutator.
#[cfg_attr(feature = "python", pyclass(name = "Mutator"))]
pub struct PyMutator {
    inner: Mutator,
}

#[cfg(feature = "python")]
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
        let inner = Mutator::new(mutation_rate, seed).map_err(PyErr::from)?;
        Ok(Self { inner })
    }

    /// Create a mutator for RNA sequences.
    #[staticmethod]
    #[pyo3(signature = (mutation_rate, seed=None))]
    fn for_rna(mutation_rate: f64, seed: Option<u64>) -> PyResult<Self> {
        let inner = Mutator::for_rna(mutation_rate, seed).map_err(PyErr::from)?;
        Ok(Self { inner })
    }

    /// Apply mutation to a sequence.
    fn apply(&mut self, sequence: &[u8]) -> Vec<u8> {
        self.inner.apply(sequence)
    }

    /// Apply mutation to multiple sequences.
    fn apply_batch(&mut self, py: Python, sequences: Vec<Vec<u8>>) -> Vec<Vec<u8>> {
        // Release GIL for parallel processing with Rayon
        py.detach(|| self.inner.apply_batch(&sequences))
    }

    fn __repr__(&self) -> String {
        "Mutator()".to_string()
    }
}

/// Python wrapper for Sampler.
#[cfg_attr(feature = "python", pyclass(name = "Sampler"))]
pub struct PySampler {
    inner: Sampler,
}

#[cfg(feature = "python")]
#[pymethods]
impl PySampler {
    /// Create a new sampler.
    ///
    /// # Arguments
    ///
    /// * `length` - Length of subsequences to extract
    /// * `strategy` - Sampling strategy: "start", "center", "end", or "random"
    /// * `seed` - Optional seed for reproducible random sampling
    #[new]
    #[pyo3(signature = (length, strategy, seed=None))]
    fn new(length: usize, strategy: &str, seed: Option<u64>) -> PyResult<Self> {
        let inner = match strategy {
            "start" => Sampler::from_start(length),
            "center" => Sampler::from_center(length),
            "end" => Sampler::from_end(length),
            "random" => Sampler::random(length, seed),
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Invalid strategy '{}'. Must be one of: 'start', 'center', 'end', 'random'",
                    strategy
                )))
            }
        };

        Ok(Self { inner })
    }

    /// Apply sampling to a sequence.
    fn apply(&mut self, sequence: &[u8]) -> Vec<u8> {
        self.inner.apply(sequence)
    }

    fn __repr__(&self) -> String {
        format!("Sampler({:?})", self.inner)
    }
}

/// Python wrapper for QualityModel.
#[cfg_attr(feature = "python", pyclass(name = "QualityModel"))]
#[derive(Clone)]
pub struct PyQualityModel {
    inner: QualityModel,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyQualityModel {
    /// Create a uniform quality distribution.
    #[staticmethod]
    #[pyo3(signature = (min, max))]
    fn uniform(min: u8, max: u8) -> Self {
        Self {
            inner: QualityModel::Uniform { min, max },
        }
    }

    /// Create a normal quality distribution.
    #[staticmethod]
    #[pyo3(signature = (mean, std_dev))]
    fn normal(mean: f64, std_dev: f64) -> Self {
        Self {
            inner: QualityModel::Normal { mean, std_dev },
        }
    }

    /// High quality preset (modern Illumina, mean ~37, std ~2).
    #[staticmethod]
    fn high_quality() -> Self {
        Self {
            inner: QualityModel::HighQuality,
        }
    }

    /// Medium quality preset (older platforms, mean ~28, std ~5).
    #[staticmethod]
    fn medium_quality() -> Self {
        Self {
            inner: QualityModel::MediumQuality,
        }
    }

    /// Degrading quality model (quality decreases along read).
    #[staticmethod]
    #[pyo3(signature = (start_mean, end_mean, std_dev))]
    fn degrading(start_mean: f64, end_mean: f64, std_dev: f64) -> Self {
        Self {
            inner: QualityModel::Degrading {
                start_mean,
                end_mean,
                std_dev,
            },
        }
    }

    fn __repr__(&self) -> String {
        match &self.inner {
            QualityModel::Uniform { min, max } => {
                format!("QualityModel.Uniform(min={}, max={})", min, max)
            }
            QualityModel::Normal { mean, std_dev } => {
                format!("QualityModel.Normal(mean={}, std_dev={})", mean, std_dev)
            }
            QualityModel::HighQuality => "QualityModel.HighQuality".to_string(),
            QualityModel::MediumQuality => "QualityModel.MediumQuality".to_string(),
            QualityModel::Degrading {
                start_mean,
                end_mean,
                std_dev,
            } => {
                format!(
                    "QualityModel.Degrading(start_mean={}, end_mean={}, std_dev={})",
                    start_mean, end_mean, std_dev
                )
            }
        }
    }
}

/// Python wrapper for QualitySimulator.
#[cfg_attr(feature = "python", pyclass(name = "QualitySimulator"))]
pub struct PyQualitySimulator {
    inner: QualitySimulator,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyQualitySimulator {
    /// Create a new quality simulator.
    ///
    /// # Arguments
    ///
    /// * `model` - QualityModel instance
    /// * `seed` - Optional seed for reproducible quality scores
    #[new]
    #[pyo3(signature = (model, seed=None))]
    fn new(model: PyQualityModel, seed: Option<u64>) -> Self {
        Self {
            inner: QualitySimulator::new(model.inner, seed),
        }
    }

    /// Generate quality scores for a sequence of given length.
    ///
    /// Returns ASCII quality scores (Phred+33 encoding).
    ///
    /// # Arguments
    ///
    /// * `length` - Number of quality scores to generate
    fn generate(&mut self, length: usize) -> Vec<u8> {
        self.inner.generate(length)
    }

    fn __repr__(&self) -> String {
        "QualitySimulator()".to_string()
    }
}

/// Register augmentation classes with Python module.
#[cfg(feature = "python")]
pub fn register_augmentation_classes(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyReverseComplement>()?;
    m.add_class::<PyMutator>()?;
    m.add_class::<PySampler>()?;
    m.add_class::<PyQualityModel>()?;
    m.add_class::<PyQualitySimulator>()?;

    // Add constants for convenience
    let quality_model_class = m.getattr("QualityModel")?;
    quality_model_class.setattr("HighQuality", PyQualityModel::high_quality())?;
    quality_model_class.setattr("MediumQuality", PyQualityModel::medium_quality())?;

    Ok(())
}
