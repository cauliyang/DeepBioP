//! Transform classes for data preprocessing.
//!
//! This module provides wrappers around existing DeepBioP encoders and
//! augmentations, exposing them with a PyTorch-compatible Transform interface.

use pyo3::prelude::*;
use pyo3::types::PyDict;

// Import existing encoders from deepbiop-fq and deepbiop-core
use deepbiop_core::kmer::encode::python::PyKmerEncoder as InnerKmerEncoder;
use deepbiop_fq::encode::integer::python::PyIntegerEncoder as InnerIntegerEncoder;
use deepbiop_fq::encode::onehot::python::PyOneHotEncoder as InnerOneHotEncoder;

// Import augmentation classes from deepbiop-fq (Rust types, not Python wrappers)
use deepbiop_fq::augment::{
    Augmentation, Mutator as InnerMutator, ReverseComplement as InnerReverseComplement,
    Sampler as InnerSampler,
};

// Transform implementations for T018-T022
// Planned classes:
// - OneHotEncoder: Wraps existing one-hot encoder (T018)
// - IntegerEncoder: Wraps existing integer encoder (T020)
// - KmerEncoder: Wraps existing k-mer encoder (T022)
// - Mutator: Wraps existing mutation augmentation (T038)
// - Sampler: Wraps existing sampling augmentation (T042)
// - ReverseComplement: Wraps existing RC augmentation (T036)
// - Compose: Simple transform chaining (<20 lines) (T036)

/// PyTorch-compatible OneHotEncoder transform.
///
/// Wraps deepbiop.fq.OneHotEncoder to provide Transform protocol:
/// - __call__(sample) -> sample with encoded sequence
/// - __repr__() for debugging
#[pyclass(name = "OneHotEncoder", module = "deepbiop.pytorch")]
pub struct OneHotEncoder {
    /// Inner encoder from deepbiop-fq
    inner: Py<InnerOneHotEncoder>,
}

#[pymethods]
impl OneHotEncoder {
    /// Create a new OneHotEncoder.
    ///
    /// Args:
    ///     encoding_type: Sequence type ("dna", "rna", "protein") - default: "dna"
    ///     unknown_strategy: How to handle unknown bases ("skip", "mask", "random") - default: "skip"
    ///     seed: Optional random seed for reproducible random replacements - default: None
    ///
    /// Returns:
    ///     OneHotEncoder instance
    ///
    /// Raises:
    ///     ValueError: If encoding_type or unknown_strategy is invalid
    ///
    /// Examples:
    ///     >>> encoder = OneHotEncoder()
    ///     >>> encoder = OneHotEncoder(encoding_type="rna", unknown_strategy="mask", seed=42)
    #[new]
    #[pyo3(signature = (encoding_type="dna", unknown_strategy="skip", seed=None))]
    fn new(
        py: Python,
        encoding_type: &str,
        unknown_strategy: &str,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        // Create inner encoder from deepbiop-fq
        let inner = Py::new(
            py,
            InnerOneHotEncoder::new(encoding_type, unknown_strategy, seed)?,
        )?;

        Ok(OneHotEncoder { inner })
    }

    /// Apply one-hot encoding to sample['sequence'].
    ///
    /// Args:
    ///     sample: Sample dict with 'sequence' key (bytes)
    ///
    /// Returns:
    ///     Sample dict with 'sequence' encoded as NumPy array
    ///
    /// Raises:
    ///     ValueError: If sample missing 'sequence' key or encoding fails
    fn __call__(&self, sample: &Bound<'_, PyDict>, py: Python) -> PyResult<Py<PyDict>> {
        // Get sequence from sample
        let sequence = sample.get_item("sequence")?.ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("Sample missing 'sequence' key")
        })?;

        // Convert sequence to Vec<u8>
        let seq_bytes: Vec<u8> = sequence.extract()?;

        // Encode using inner encoder
        let inner_encoder = self.inner.borrow(py);
        let encoded = inner_encoder.encode(py, seq_bytes)?;

        // Create new sample dict with encoded sequence
        let new_sample = PyDict::new(py);
        new_sample.set_item("sequence", encoded)?;

        // Copy other keys from original sample
        for (key, value) in sample.iter() {
            let key_str: String = key.extract()?;
            if key_str != "sequence" {
                new_sample.set_item(key, value)?;
            }
        }

        Ok(new_sample.into())
    }

    /// Human-readable representation.
    fn __repr__(&self, py: Python) -> PyResult<String> {
        let inner_encoder = self.inner.borrow(py);
        let encoding_type = inner_encoder.encoding_type();
        let unknown_strategy = inner_encoder.ambiguous_strategy();

        Ok(format!(
            "OneHotEncoder(encoding_type='{}', unknown_strategy='{}')",
            encoding_type, unknown_strategy
        ))
    }
}

/// PyTorch-compatible IntegerEncoder transform.
///
/// Wraps deepbiop.fq.IntegerEncoder to provide Transform protocol:
/// - __call__(sample) -> sample with encoded sequence
/// - __repr__() for debugging
#[pyclass(name = "IntegerEncoder", module = "deepbiop.pytorch")]
pub struct IntegerEncoder {
    /// Inner encoder from deepbiop-fq
    inner: Py<InnerIntegerEncoder>,
}

#[pymethods]
impl IntegerEncoder {
    /// Create a new IntegerEncoder.
    ///
    /// Args:
    ///     encoding_type: Sequence type ("dna", "rna", "protein")
    ///
    /// Returns:
    ///     IntegerEncoder instance
    ///
    /// Raises:
    ///     ValueError: If encoding_type is invalid
    #[new]
    #[pyo3(signature = (encoding_type="dna"))]
    fn new(py: Python, encoding_type: &str) -> PyResult<Self> {
        // Create inner encoder from deepbiop-fq
        let inner = Py::new(py, InnerIntegerEncoder::new(encoding_type)?)?;
        Ok(IntegerEncoder { inner })
    }

    /// Apply integer encoding to sample['sequence'].
    ///
    /// Args:
    ///     sample: Sample dict with 'sequence' key (bytes)
    ///
    /// Returns:
    ///     Sample dict with 'sequence' encoded as NumPy array
    ///
    /// Raises:
    ///     ValueError: If sample missing 'sequence' key or encoding fails
    fn __call__(&self, sample: &Bound<'_, PyDict>, py: Python) -> PyResult<Py<PyDict>> {
        // Get sequence from sample
        let sequence = sample.get_item("sequence")?.ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("Sample missing 'sequence' key")
        })?;

        // Convert sequence to Vec<u8>
        let seq_bytes: Vec<u8> = sequence.extract()?;

        // Encode using inner encoder
        let inner_encoder = self.inner.borrow(py);
        let encoded = inner_encoder.encode(py, seq_bytes)?;

        // Create new sample dict with encoded sequence
        let new_sample = PyDict::new(py);
        new_sample.set_item("sequence", encoded)?;

        // Copy other keys from original sample
        for (key, value) in sample.iter() {
            let key_str: String = key.extract()?;
            if key_str != "sequence" {
                new_sample.set_item(key, value)?;
            }
        }

        Ok(new_sample.into())
    }

    /// Human-readable representation.
    fn __repr__(&self, py: Python) -> PyResult<String> {
        let inner_encoder = self.inner.borrow(py);
        let encoding_type = inner_encoder.encoding_type();

        Ok(format!("IntegerEncoder(encoding_type='{}')", encoding_type))
    }
}

/// PyTorch-compatible KmerEncoder transform.
///
/// Wraps deepbiop.core.KmerEncoder to provide Transform protocol:
/// - __call__(sample) -> sample with encoded sequence
/// - __repr__() for debugging
#[pyclass(name = "KmerEncoder", module = "deepbiop.pytorch")]
pub struct KmerEncoder {
    /// Inner encoder from deepbiop-core
    inner: Py<InnerKmerEncoder>,
}

#[pymethods]
impl KmerEncoder {
    /// Create a new KmerEncoder.
    ///
    /// Args:
    ///     k: K-mer length
    ///     canonical: Whether to use canonical k-mers
    ///     encoding_type: Sequence type ("dna", "rna", "protein")
    ///
    /// Returns:
    ///     KmerEncoder instance
    ///
    /// Raises:
    ///     ValueError: If k or encoding_type is invalid
    #[new]
    #[pyo3(signature = (k=3, canonical=false, encoding_type="dna"))]
    fn new(py: Python, k: usize, canonical: bool, encoding_type: &str) -> PyResult<Self> {
        // Create inner encoder from deepbiop-core
        let inner = Py::new(py, InnerKmerEncoder::new(k, canonical, encoding_type)?)?;
        Ok(KmerEncoder { inner })
    }

    /// Apply k-mer encoding to sample['sequence'].
    ///
    /// Args:
    ///     sample: Sample dict with 'sequence' key (bytes)
    ///
    /// Returns:
    ///     Sample dict with 'sequence' encoded as NumPy array (k-mer frequency vector)
    ///
    /// Raises:
    ///     ValueError: If sample missing 'sequence' key or encoding fails
    fn __call__(&mut self, sample: &Bound<'_, PyDict>, py: Python) -> PyResult<Py<PyDict>> {
        // Get sequence from sample
        let sequence = sample.get_item("sequence")?.ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("Sample missing 'sequence' key")
        })?;

        // Convert sequence to Vec<u8>
        let seq_bytes: Vec<u8> = sequence.extract()?;

        // Encode using inner encoder (note: need mutable borrow)
        let mut inner_encoder = self.inner.borrow_mut(py);
        let encoded = inner_encoder.encode(py, seq_bytes)?;

        // Create new sample dict with encoded sequence
        let new_sample = PyDict::new(py);
        new_sample.set_item("sequence", encoded)?;

        // Copy other keys from original sample
        for (key, value) in sample.iter() {
            let key_str: String = key.extract()?;
            if key_str != "sequence" {
                new_sample.set_item(key, value)?;
            }
        }

        Ok(new_sample.into())
    }

    /// Human-readable representation.
    fn __repr__(&self, py: Python) -> PyResult<String> {
        let inner_encoder = self.inner.borrow(py);
        let k = inner_encoder.k();
        let canonical = inner_encoder.is_canonical();
        let encoding_type = inner_encoder.encoding_type();

        Ok(format!(
            "KmerEncoder(k={}, canonical={}, encoding_type='{}')",
            k, canonical, encoding_type
        ))
    }
}

/// Compose multiple transforms sequentially.
///
/// Chains transforms by calling them in order on the sample.
/// Similar to torchvision.transforms.Compose.
#[pyclass(name = "Compose", module = "deepbiop.pytorch")]
pub struct Compose {
    /// List of transforms to apply sequentially
    transforms: Vec<Py<pyo3::PyAny>>,
}

#[pymethods]
impl Compose {
    /// Create a new Compose transform.
    ///
    /// Args:
    ///     transforms: List of transform callables
    ///
    /// Returns:
    ///     Compose instance
    #[new]
    fn new(transforms: Vec<Py<pyo3::PyAny>>) -> Self {
        Compose { transforms }
    }

    /// Apply all transforms sequentially.
    ///
    /// Args:
    ///     sample: Sample dict
    ///
    /// Returns:
    ///     Transformed sample after applying all transforms
    fn __call__(&self, mut sample: Py<pyo3::PyAny>, py: Python) -> PyResult<Py<pyo3::PyAny>> {
        for transform in &self.transforms {
            sample = transform.call1(py, (sample,))?;
        }
        Ok(sample)
    }

    /// Human-readable representation.
    fn __repr__(&self) -> String {
        format!("Compose({} transforms)", self.transforms.len())
    }
}

/// PyTorch-compatible ReverseComplement transform.
///
/// Wraps deepbiop.fq.ReverseComplement to provide Transform protocol.
#[pyclass(name = "ReverseComplement", module = "deepbiop.pytorch")]
pub struct ReverseComplement {
    /// Inner augmentation from deepbiop-fq
    inner: InnerReverseComplement,
}

#[pymethods]
impl ReverseComplement {
    /// Create a new ReverseComplement transform.
    ///
    /// Returns:
    ///     ReverseComplement instance
    #[new]
    fn new() -> Self {
        ReverseComplement {
            inner: InnerReverseComplement::new(),
        }
    }

    /// Apply reverse complement to sample['sequence'].
    ///
    /// Args:
    ///     sample: Sample dict with 'sequence' key (bytes)
    ///
    /// Returns:
    ///     Sample dict with reverse complemented sequence
    fn __call__(&mut self, sample: &Bound<'_, PyDict>, py: Python) -> PyResult<Py<PyDict>> {
        // Get sequence from sample
        let sequence = sample.get_item("sequence")?.ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("Sample missing 'sequence' key")
        })?;

        // Convert sequence to Vec<u8>
        let seq_bytes: Vec<u8> = sequence.extract()?;

        // Apply reverse complement using inner transform
        let rc_bytes = self.inner.apply(&seq_bytes);

        // Create new sample dict with transformed sequence
        let new_sample = PyDict::new(py);
        new_sample.set_item("sequence", pyo3::types::PyBytes::new(py, &rc_bytes))?;

        // Copy other keys (reverse quality if present)
        for (key, value) in sample.iter() {
            let key_str: String = key.extract()?;
            if key_str == "quality" {
                // Reverse quality scores to match reversed sequence
                let qual_bytes: Vec<u8> = value.extract()?;
                let mut reversed_qual = qual_bytes;
                reversed_qual.reverse();
                new_sample.set_item(key, pyo3::types::PyBytes::new(py, &reversed_qual))?;
            } else if key_str != "sequence" {
                new_sample.set_item(key, value)?;
            }
        }

        Ok(new_sample.into())
    }

    /// Human-readable representation.
    fn __repr__(&self) -> String {
        "ReverseComplement()".to_string()
    }
}

/// PyTorch-compatible Mutator transform.
///
/// Wraps deepbiop.fq.Mutator to provide Transform protocol.
#[pyclass(name = "Mutator", module = "deepbiop.pytorch")]
pub struct Mutator {
    /// Inner augmentation from deepbiop-fq
    inner: InnerMutator,
    mutation_rate: f64,
}

#[pymethods]
impl Mutator {
    /// Create a new Mutator transform.
    ///
    /// Args:
    ///     mutation_rate: Probability of mutation per base
    ///     seed: Optional random seed for reproducibility
    ///
    /// Returns:
    ///     Mutator instance
    #[new]
    #[pyo3(signature = (mutation_rate, seed=None))]
    fn new(mutation_rate: f64, seed: Option<u64>) -> Self {
        Mutator {
            inner: InnerMutator::new(mutation_rate, seed),
            mutation_rate,
        }
    }

    /// Apply mutation to sample['sequence'].
    ///
    /// Args:
    ///     sample: Sample dict with 'sequence' key (bytes)
    ///
    /// Returns:
    ///     Sample dict with mutated sequence
    fn __call__(&mut self, sample: &Bound<'_, PyDict>, py: Python) -> PyResult<Py<PyDict>> {
        // Get sequence from sample
        let sequence = sample.get_item("sequence")?.ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("Sample missing 'sequence' key")
        })?;

        // Convert sequence to Vec<u8>
        let seq_bytes: Vec<u8> = sequence.extract()?;

        // Apply mutation using inner transform
        let mutated_bytes = self.inner.apply(&seq_bytes);

        // Create new sample dict with transformed sequence
        let new_sample = PyDict::new(py);
        new_sample.set_item("sequence", pyo3::types::PyBytes::new(py, &mutated_bytes))?;

        // Copy other keys from original sample
        for (key, value) in sample.iter() {
            let key_str: String = key.extract()?;
            if key_str != "sequence" {
                new_sample.set_item(key, value)?;
            }
        }

        Ok(new_sample.into())
    }

    /// Human-readable representation.
    fn __repr__(&self) -> String {
        format!("Mutator(mutation_rate={})", self.mutation_rate)
    }
}

/// PyTorch-compatible Sampler transform.
///
/// Wraps deepbiop.fq.Sampler to provide Transform protocol.
#[pyclass(name = "Sampler", module = "deepbiop.pytorch")]
pub struct Sampler {
    /// Inner augmentation from deepbiop-fq
    inner: InnerSampler,
    length: usize,
    strategy: String,
}

#[pymethods]
impl Sampler {
    /// Create a new Sampler transform.
    ///
    /// Args:
    ///     length: Target length to sample
    ///     strategy: Sampling strategy ("start", "center", "end", "random")
    ///     seed: Optional random seed for reproducible random sampling
    ///
    /// Returns:
    ///     Sampler instance
    #[new]
    #[pyo3(signature = (length, strategy="start", seed=None))]
    fn new(length: usize, strategy: &str, seed: Option<u64>) -> PyResult<Self> {
        let inner = match strategy {
            "start" => InnerSampler::from_start(length),
            "center" => InnerSampler::from_center(length),
            "end" => InnerSampler::from_end(length),
            "random" => InnerSampler::random(length, seed),
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Invalid strategy '{}'. Must be one of: 'start', 'center', 'end', 'random'",
                    strategy
                )))
            }
        };

        Ok(Sampler {
            inner,
            length,
            strategy: strategy.to_string(),
        })
    }

    /// Apply sampling to sample['sequence'].
    ///
    /// Args:
    ///     sample: Sample dict with 'sequence' key (bytes)
    ///
    /// Returns:
    ///     Sample dict with sampled sequence
    fn __call__(&mut self, sample: &Bound<'_, PyDict>, py: Python) -> PyResult<Py<PyDict>> {
        // Get sequence from sample
        let sequence = sample.get_item("sequence")?.ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("Sample missing 'sequence' key")
        })?;

        // Convert sequence to Vec<u8>
        let seq_bytes: Vec<u8> = sequence.extract()?;

        // Apply sampling using inner transform
        let sampled_bytes = self.inner.apply(&seq_bytes);

        // Create new sample dict with transformed sequence
        let new_sample = PyDict::new(py);
        new_sample.set_item("sequence", pyo3::types::PyBytes::new(py, &sampled_bytes))?;

        // Copy other keys (sample quality if present and same strategy)
        for (key, value) in sample.iter() {
            let key_str: String = key.extract()?;
            if key_str == "quality" {
                // Sample quality scores to match sampled sequence
                let qual_bytes: Vec<u8> = value.extract()?;
                let sampled_qual = self.inner.apply(&qual_bytes);
                new_sample.set_item(key, pyo3::types::PyBytes::new(py, &sampled_qual))?;
            } else if key_str != "sequence" {
                new_sample.set_item(key, value)?;
            }
        }

        Ok(new_sample.into())
    }

    /// Human-readable representation.
    fn __repr__(&self) -> String {
        format!(
            "Sampler(length={}, strategy='{}')",
            self.length, self.strategy
        )
    }
}
