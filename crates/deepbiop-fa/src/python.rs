use std::path::PathBuf;

use crate::encode::Encoder;
use crate::{
    encode::{self},
    io,
};

use ahash::HashSet;
use anyhow::Result;
use bstr::BString;
#[cfg(feature = "cache")]
use deepbiop_utils::io as deepbiop_io;
use log::warn;
use noodles::fastq;
use numpy::convert::ToPyArray;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;

use deepbiop_core::dataset::IterableDataset;
use pyo3_stub_gen::derive::*;

#[gen_stub_pymethods]
#[pymethods]
impl encode::ParquetEncoder {
    #[new]
    fn py_new(option: encode::EncoderOption) -> Self {
        encode::ParquetEncoder::new(option)
    }
}

#[gen_stub_pyclass]
#[pyclass(name = "RecordData")]
pub struct PyRecordData(encode::RecordData);

impl From<encode::RecordData> for PyRecordData {
    fn from(data: encode::RecordData) -> Self {
        Self(data)
    }
}

// Implement FromPyObject for PyRecordData
impl<'a, 'py> FromPyObject<'a, 'py> for PyRecordData {
    type Error = PyErr;

    fn extract(ob: pyo3::Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        // Assuming Python objects are tuples of (id, seq)
        let (id, seq): (String, String) = ob.extract()?;
        Ok(PyRecordData(encode::RecordData {
            id: id.into(),
            seq: seq.into(),
        }))
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyRecordData {
    #[new]
    fn new(id: String, seq: String) -> Self {
        Self(encode::RecordData {
            id: id.into(),
            seq: seq.into(),
        })
    }

    #[getter]
    fn id(&self) -> String {
        self.0.id.to_string()
    }

    #[setter]
    fn set_id(&mut self, id: String) {
        self.0.id = id.into();
    }

    #[getter]
    fn seq(&self) -> String {
        self.0.seq.to_string()
    }

    #[setter]
    fn set_seq(&mut self, seq: String) {
        self.0.seq = seq.into();
    }
}

#[gen_stub_pyfunction()]
#[pyfunction]
#[pyo3(signature = (records_data, file_path=None))]
fn write_fa(records_data: Vec<PyRecordData>, file_path: Option<PathBuf>) -> Result<()> {
    let records: Vec<encode::RecordData> = records_data
        .into_par_iter()
        .map(|py_record| py_record.0)
        .collect();
    io::write_fa(&records, file_path)
}

#[gen_stub_pyfunction()]
#[pyfunction]
fn write_fa_parallel(
    records_data: Vec<PyRecordData>,
    file_path: PathBuf,
    threads: usize,
) -> Result<()> {
    let records: Vec<encode::RecordData> = records_data
        .into_par_iter()
        .map(|py_record| py_record.0)
        .collect();
    io::write_bzip_fa_parallel(&records, file_path, Some(threads))
}

#[cfg(feature = "cache")]
#[gen_stub_pyfunction()]
#[pyfunction]
fn encode_fa_path_to_parquet_chunk(
    fa_path: PathBuf,
    chunk_size: usize,
    parallel: bool,
    bases: String,
) -> Result<()> {
    let option = encode::EncoderOptionBuilder::default()
        .bases(bases.as_bytes().to_vec())
        .build()?;

    let mut fa_encoder = encode::ParquetEncoderBuilder::default()
        .option(option)
        .build()?;
    fa_encoder.encode_chunk(&fa_path, chunk_size, parallel)?;
    Ok(())
}

#[gen_stub_pyfunction()]
#[pyfunction]
#[pyo3(signature = (fa_path, bases, result_path=None))]
fn encode_fa_path_to_parquet(
    fa_path: PathBuf,
    bases: String,
    result_path: Option<PathBuf>,
) -> Result<()> {
    let option = encode::EncoderOptionBuilder::default()
        .bases(bases.as_bytes().to_vec())
        .build()?;

    let mut fa_encoder = encode::ParquetEncoderBuilder::default()
        .option(option)
        .build()?;
    let (record_batch, schema) = fa_encoder.encode(&fa_path)?;

    // result file is fq_path with .parquet extension
    let parquet_path = if let Some(path) = result_path {
        if path.with_extension("parquet").exists() {
            warn!("{} already exists, overwriting", path.display());
        }
        path.with_extension("parquet")
    } else {
        fa_path.with_extension("parquet")
    };

    #[cfg(feature = "cache")]
    {
        deepbiop_io::write_parquet_for_batches(parquet_path, &record_batch, schema)?;
        Ok(())
    }

    #[cfg(not(feature = "cache"))]
    {
        let _ = (parquet_path, record_batch, schema); // Silence unused warnings
        Err(anyhow::anyhow!("Parquet encoding requires 'cache' feature"))
    }
}

#[gen_stub_pyfunction()]
#[pyfunction]
fn encode_fa_paths_to_parquet(fa_path: Vec<PathBuf>, bases: String) -> Result<()> {
    fa_path.iter().for_each(|path| {
        encode_fa_path_to_parquet(path.clone(), bases.clone(), None).unwrap();
    });
    Ok(())
}

#[gen_stub_pyfunction()]
#[pyfunction]
fn convert_multiple_fas_to_one_fa(
    paths: Vec<PathBuf>,
    result_path: PathBuf,
    parallel: bool,
) -> Result<()> {
    if paths.is_empty() {
        return Ok(());
    }
    io::convert_multiple_fas_to_one_bgzip_fa(&paths, result_path, parallel)?;
    Ok(())
}

#[gen_stub_pyfunction()]
#[pyfunction(name = "select_record_from_fa")]
pub fn py_select_record_from_fq(
    selected_reads: Vec<String>,
    fq: PathBuf,
    output: PathBuf,
) -> Result<()> {
    let selected_reads: HashSet<BString> =
        selected_reads.into_par_iter().map(|s| s.into()).collect();
    let records = io::select_record_from_fa_by_stream(fq, &selected_reads)?;
    io::write_fa_for_noodle_record(&records, output)?;
    Ok(())
}

#[gen_stub_pyfunction()]
#[pyfunction(name = "select_record_from_fa_by_random")]
pub fn py_select_record_from_fq_by_random(
    fq: PathBuf,
    number: usize,
    output: PathBuf,
) -> Result<()> {
    let records = io::select_record_from_fa_by_random(fq, number)?;
    io::write_fa_for_noodle_record(&records, output)?;
    Ok(())
}

/// Convert FASTA file to FASTQ file with default quality scores.
///
/// Since FASTA files don't contain quality information, assigns
/// default quality score (Phred+33 Q40 = '~') to all bases.
#[gen_stub_pyfunction()]
#[pyfunction]
pub fn fasta_to_fastq(fasta_path: PathBuf, fastq_path: PathBuf) -> Result<()> {
    let fq_records = io::fasta_to_fastq(&fasta_path)?;
    let handle = std::fs::File::create(fastq_path)?;
    let mut writer = fastq::io::Writer::new(handle);
    for record in fq_records {
        writer.write_record(&record)?;
    }
    Ok(())
}

/// Streaming FASTA dataset for efficient iteration over large files.
///
/// This dataset provides memory-efficient streaming iteration over FASTA files,
/// reading records one at a time without loading the entire file into memory.
/// Yields records as dictionaries with NumPy arrays for zero-copy efficiency.
///
/// Note: FASTA files do not contain quality scores.
///
/// # Examples
///
/// ```python
/// from deepbiop.fa import FastaStreamDataset
///
/// dataset = FastaStreamDataset("genome.fasta.gz")
/// for record in dataset:
///     seq = record['sequence']  # NumPy array
///     print(f"ID: {record['id']}, Length: {len(seq)}")
/// ```
#[gen_stub_pyclass]
#[pyclass(name = "FastaStreamDataset", module = "deepbiop.fa")]
pub struct PyFastaStreamDataset {
    file_path: String,
    size_hint: Option<usize>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyFastaStreamDataset {
    /// Create a new streaming FASTA dataset.
    ///
    /// # Arguments
    ///
    /// * `file_path` - Path to FASTA file (supports .fasta, .fa, .fasta.gz, .fa.gz)
    ///
    /// # Returns
    ///
    /// FastaStreamDataset instance
    ///
    /// # Raises
    ///
    /// * `FileNotFoundError` - If file doesn't exist
    /// * `IOError` - If file cannot be opened
    #[new]
    fn new(file_path: String) -> PyResult<Self> {
        // Create temporary dataset to validate file and get size hint
        let dataset = crate::dataset::FastaDataset::new(file_path.clone())
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

        let size_hint = dataset.records_count();

        Ok(Self {
            file_path,
            size_hint,
        })
    }

    /// Return iterator over dataset records.
    ///
    /// Each record is a dictionary with:
    /// - 'id': str - Sequence identifier
    /// - 'sequence': np.ndarray (uint8) - Nucleotide sequence bytes
    /// - 'description': Optional[str] - Sequence description
    fn __iter__(slf: PyRef<'_, Self>) -> PyResult<Py<PyFastaStreamIterator>> {
        let iter = PyFastaStreamIterator {
            file_path: slf.file_path.clone(),
            current_idx: 0,
        };
        Py::new(slf.py(), iter)
    }

    /// Get estimated number of records in dataset.
    ///
    /// # Returns
    ///
    /// int - Estimated record count (0 if unavailable)
    fn __len__(&self) -> PyResult<usize> {
        Ok(self.size_hint.unwrap_or(0))
    }

    /// Human-readable representation.
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "FastaStreamDataset(file='{}', records={})",
            self.file_path,
            self.size_hint
                .map_or("unknown".to_string(), |n| n.to_string())
        ))
    }

    /// Pickling support for multiprocessing (DataLoader with num_workers > 0).
    fn __getstate__(&self, py: Python) -> PyResult<Py<PyDict>> {
        let state = PyDict::new(py);
        state.set_item("file_path", &self.file_path)?;
        state.set_item("size_hint", self.size_hint)?;
        Ok(state.into())
    }

    /// Unpickling support for multiprocessing.
    fn __setstate__(&mut self, state: &Bound<'_, PyDict>) -> PyResult<()> {
        self.file_path = state.get_item("file_path")?.unwrap().extract()?;
        self.size_hint = state.get_item("size_hint")?.unwrap().extract()?;
        Ok(())
    }
}

/// Iterator for streaming FASTA dataset.
#[gen_stub_pyclass]
#[pyclass(name = "FastaStreamIterator", module = "deepbiop.fa")]
pub struct PyFastaStreamIterator {
    file_path: String,
    current_idx: usize,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyFastaStreamIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python) -> PyResult<Option<Py<PyDict>>> {
        // Create a new dataset and iterator for each record
        let dataset = crate::dataset::FastaDataset::new(self.file_path.clone())
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

                // Convert sequence to NumPy array (zero-copy from Rust Vec)
                let seq_array = record.sequence.to_pyarray(py);
                dict.set_item("sequence", seq_array)?;

                // FASTA files don't have quality scores
                dict.set_item("quality", py.None())?;

                dict.set_item("description", record.description)?;

                Ok(Some(dict.into()))
            }
            Some(Err(e)) => Err(pyo3::exceptions::PyIOError::new_err(format!(
                "Failed to read FASTA record: {}",
                e
            ))),
        }
    }
}

// register fq sub_module
pub fn register_fa_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let sub_module_name = "fa";
    let child_module = PyModule::new(parent_module.py(), sub_module_name)?;

    child_module.add_class::<PyRecordData>()?;
    child_module.add_class::<encode::EncoderOption>()?;
    child_module.add_class::<encode::ParquetEncoder>()?;

    child_module.add_function(wrap_pyfunction!(write_fa, &child_module)?)?;
    child_module.add_function(wrap_pyfunction!(write_fa_parallel, &child_module)?)?;
    child_module.add_function(wrap_pyfunction!(encode_fa_path_to_parquet, &child_module)?)?;
    child_module.add_function(wrap_pyfunction!(encode_fa_paths_to_parquet, &child_module)?)?;
    #[cfg(feature = "cache")]
    child_module.add_function(wrap_pyfunction!(
        encode_fa_path_to_parquet_chunk,
        &child_module
    )?)?;
    child_module.add_function(wrap_pyfunction!(
        convert_multiple_fas_to_one_fa,
        &child_module
    )?)?;

    child_module.add_function(wrap_pyfunction!(py_select_record_from_fq, &child_module)?)?;
    child_module.add_function(wrap_pyfunction!(
        py_select_record_from_fq_by_random,
        &child_module
    )?)?;
    child_module.add_function(wrap_pyfunction!(fasta_to_fastq, &child_module)?)?;

    // Add streaming dataset classes
    child_module.add_class::<PyFastaStreamDataset>()?;
    child_module.add_class::<PyFastaStreamIterator>()?;

    parent_module.add_submodule(&child_module)?;

    Ok(())
}
