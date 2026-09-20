//! Dataset class for PyTorch-style data loading.
//!
//! This module provides the Dataset class that wraps DeepBioP file readers
//! and presents a PyTorch-compatible map-style interface (`__len__`,
//! `__getitem__`, `__iter__`) for FASTQ files.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3_stub_gen::derive::*;
use std::fs::File;
use std::io::{BufReader, Seek, SeekFrom};
use std::path::Path;
use std::sync::Mutex;

use deepbiop_utils::io::{
    check_compressed_type, create_reader_for_compressed_file, CompressedType,
};
use noodles::fastq;

/// Record storage backing random access.
///
/// Plain files are indexed by byte offset and read on demand, so memory is
/// eight bytes per record. Compressed inputs cannot be seeked, so they are
/// decompressed once into memory.
enum Storage {
    Indexed {
        reader: Mutex<fastq::io::Reader<BufReader<File>>>,
        offsets: Vec<u64>,
    },
    InMemory(Vec<fastq::Record>),
}

impl Storage {
    fn open(file_path: &str) -> anyhow::Result<Self> {
        let is_plain = matches!(
            check_compressed_type(file_path)?,
            CompressedType::Uncompress
        );

        if !is_plain {
            let reader = BufReader::new(create_reader_for_compressed_file(file_path)?);
            let mut reader = fastq::io::Reader::new(reader);
            let records = reader.records().collect::<std::io::Result<Vec<_>>>()?;
            return Ok(Storage::InMemory(records));
        }

        let mut reader =
            fastq::io::Reader::new(BufReader::with_capacity(1 << 16, File::open(file_path)?));
        let mut offsets = Vec::new();
        let mut record = fastq::Record::default();
        loop {
            let offset = reader.get_mut().stream_position()?;
            if reader.read_record(&mut record)? == 0 {
                break;
            }
            offsets.push(offset);
        }
        Ok(Storage::Indexed {
            reader: Mutex::new(reader),
            offsets,
        })
    }

    fn len(&self) -> usize {
        match self {
            Storage::Indexed { offsets, .. } => offsets.len(),
            Storage::InMemory(records) => records.len(),
        }
    }

    /// Fetch `(sequence, quality)` for a record index. Caller checks bounds.
    fn get(&self, idx: usize) -> PyResult<(Vec<u8>, Vec<u8>)> {
        match self {
            Storage::InMemory(records) => {
                let r = &records[idx];
                Ok((r.sequence().to_vec(), r.quality_scores().to_vec()))
            }
            Storage::Indexed { reader, offsets } => {
                let mut reader = reader.lock().map_err(|_| {
                    pyo3::exceptions::PyRuntimeError::new_err("dataset reader poisoned")
                })?;
                reader
                    .get_mut()
                    .seek(SeekFrom::Start(offsets[idx]))
                    .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
                let mut record = fastq::Record::default();
                let n = reader
                    .read_record(&mut record)
                    .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
                if n == 0 {
                    return Err(pyo3::exceptions::PyIOError::new_err(format!(
                        "record {idx} vanished from file (file modified after indexing?)"
                    )));
                }
                Ok((record.sequence().to_vec(), record.quality_scores().to_vec()))
            }
        }
    }
}

/// PyTorch-compatible map-style Dataset for FASTQ files.
///
/// - `__len__()`: number of records
/// - `__getitem__(i)`: `{"sequence": bytes, "quality": bytes}` (after `transform`, if any)
/// - `__iter__()`: sequential iteration
#[gen_stub_pyclass]
#[pyclass(name = "Dataset", module = "deepbiop.pytorch", sequence)]
pub struct Dataset {
    storage: Storage,
    file_path: String,
    sequence_type: String,
    transform: Option<Py<PyAny>>,
}

#[gen_stub_pymethods]
#[pymethods]
impl Dataset {
    /// Create a new Dataset from a FASTQ file (plain, gzip or bgzip).
    ///
    /// Args:
    ///     file_path: Path to the FASTQ file
    ///     sequence_type: "dna", "rna" or "protein"; recorded for downstream encoders
    ///     transform: Optional callable applied to every sample dict returned by
    ///         `__getitem__`/iteration (e.g. an encoder or `Compose`)
    ///
    /// Raises:
    ///     FileNotFoundError: If the file does not exist
    ///     IOError: If the file cannot be parsed
    #[new]
    #[pyo3(signature = (file_path, *, sequence_type="dna", transform=None))]
    fn new(file_path: String, sequence_type: &str, transform: Option<Py<PyAny>>) -> PyResult<Self> {
        if !Path::new(&file_path).exists() {
            return Err(pyo3::exceptions::PyFileNotFoundError::new_err(format!(
                "FASTQ file '{file_path}' not found. Check path and permissions."
            )));
        }
        match sequence_type {
            "dna" | "rna" | "protein" => {}
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "sequence_type must be 'dna', 'rna' or 'protein', got '{other}'"
                )))
            }
        }

        let storage = Storage::open(&file_path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

        Ok(Dataset {
            storage,
            file_path,
            sequence_type: sequence_type.to_owned(),
            transform,
        })
    }

    /// Returns total number of sequences in dataset.
    fn __len__(&self) -> usize {
        self.storage.len()
    }

    /// Get sample at index idx.
    ///
    /// Args:
    ///     idx: Sample index (0 to len(dataset)-1)
    ///
    /// Returns:
    ///     Sample dict with 'sequence' and 'quality' keys, passed through
    ///     `transform` when one was given.
    ///
    /// Raises:
    ///     IndexError: If idx out of range
    fn __getitem__(&self, idx: usize, py: Python) -> PyResult<Py<PyAny>> {
        let len = self.storage.len();
        if idx >= len {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "Index {idx} out of range (dataset has {len} sequences)"
            )));
        }

        let (sequence, quality) = self.storage.get(idx)?;
        let sample = PyDict::new(py);
        sample.set_item("sequence", sequence.as_slice())?;
        sample.set_item("quality", quality.as_slice())?;

        match &self.transform {
            Some(transform) => transform.call1(py, (sample,)),
            None => Ok(sample.into_any().unbind()),
        }
    }

    /// Iterate over all samples in dataset.
    fn __iter__(slf: PyRef<'_, Self>) -> DatasetIterator {
        DatasetIterator {
            dataset: slf.into(),
            current_idx: 0,
        }
    }

    /// Human-readable representation.
    fn __repr__(&self, py: Python) -> PyResult<String> {
        let transform = match &self.transform {
            Some(t) => t.bind(py).repr()?.to_string(),
            None => "None".to_owned(),
        };
        Ok(format!(
            "Dataset(path='{}', num_samples={}, sequence_type='{}', transform={})",
            self.file_path,
            self.storage.len(),
            self.sequence_type,
            transform
        ))
    }

    /// Get dataset statistics.
    ///
    /// Returns:
    ///     Dict with keys:
    ///         - 'num_samples': int - Total number of sequences
    ///         - 'length_stats': dict - Statistics with 'min', 'max', 'mean', 'median' sequence lengths
    ///         - 'memory_footprint': int - Estimated memory usage in bytes (sequence + quality)
    fn summary(&self, py: Python) -> PyResult<Py<PyDict>> {
        let num_records = self.storage.len();
        if num_records == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Dataset is empty, cannot generate summary",
            ));
        }

        let mut lengths = Vec::with_capacity(num_records);
        for idx in 0..num_records {
            lengths.push(self.storage.get(idx)?.0.len());
        }

        let min_len = *lengths.iter().min().unwrap();
        let max_len = *lengths.iter().max().unwrap();
        let sum: usize = lengths.iter().sum();
        let mean_len = sum as f64 / lengths.len() as f64;

        lengths.sort_unstable();
        let median_len = if lengths.len() % 2 == 0 {
            let mid = lengths.len() / 2;
            (lengths[mid - 1] + lengths[mid]) as f64 / 2.0
        } else {
            lengths[lengths.len() / 2] as f64
        };

        let memory_footprint = sum * 2;

        let summary = PyDict::new(py);
        summary.set_item("num_samples", num_records)?;

        let length_stats = PyDict::new(py);
        length_stats.set_item("min", min_len)?;
        length_stats.set_item("max", max_len)?;
        length_stats.set_item("mean", mean_len)?;
        length_stats.set_item("median", median_len)?;
        summary.set_item("length_stats", length_stats)?;
        summary.set_item("memory_footprint", memory_footprint)?;

        Ok(summary.into())
    }

    /// Validate dataset quality and integrity on the first 10 records.
    ///
    /// Returns:
    ///     Dict with keys 'is_valid' (bool), 'warnings' (list[str]), 'errors' (list[str])
    fn validate(&self, py: Python) -> PyResult<Py<PyDict>> {
        let mut warnings: Vec<String> = Vec::new();
        let mut errors: Vec<String> = Vec::new();
        let num_records = self.storage.len();

        if num_records == 0 {
            errors.push("Dataset is empty".to_string());
        }

        let sample_size = std::cmp::min(10, num_records);
        let valid_bases = b"ACGTNacgtnUu-";

        for idx in 0..sample_size {
            let (sequence, quality) = self.storage.get(idx)?;

            if sequence.is_empty() {
                errors.push(format!("Empty sequence at index {idx}"));
            }
            if sequence.len() != quality.len() {
                warnings.push(format!(
                    "Quality length ({}) doesn't match sequence length ({}) at index {idx}",
                    quality.len(),
                    sequence.len(),
                ));
            }
            if let Some(&base) = sequence.iter().find(|b| !valid_bases.contains(b)) {
                warnings.push(format!(
                    "Unexpected base '{}' at index {idx}",
                    char::from(base)
                ));
            }
        }

        if sample_size < num_records {
            warnings.push(format!(
                "Validated {sample_size} of {num_records} sequences (sample validation)"
            ));
        }

        let result = PyDict::new(py);
        result.set_item("is_valid", errors.is_empty())?;
        result.set_item("warnings", warnings)?;
        result.set_item("errors", errors)?;
        Ok(result.into())
    }
}

// Rust-side accessors used by DataLoader.
impl Dataset {
    pub fn len(&self) -> usize {
        self.storage.len()
    }

    pub fn get_item(&self, idx: usize, py: Python) -> PyResult<Py<PyAny>> {
        self.__getitem__(idx, py)
    }
}

/// Iterator for Dataset.
#[gen_stub_pyclass]
#[pyclass(name = "DatasetIterator", module = "deepbiop.pytorch")]
pub struct DatasetIterator {
    dataset: Py<Dataset>,
    current_idx: usize,
}

#[gen_stub_pymethods]
#[pymethods]
impl DatasetIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python) -> PyResult<Option<Py<PyAny>>> {
        let dataset = self.dataset.borrow(py);
        if self.current_idx >= dataset.len() {
            return Ok(None);
        }
        let sample = dataset.__getitem__(self.current_idx, py)?;
        self.current_idx += 1;
        Ok(Some(sample))
    }
}
