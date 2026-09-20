use bstr::BString;
use std::path::PathBuf;

use crate::{
    dataset::{
        count_records_efficient, FastqDataset, FastqIterator, FastqRecord, FastqStreamIterator,
    },
    encode::{self, Encoder},
    filter, io,
    predicts::{self, Predict},
    streaming, utils,
};

use ahash::{HashMap, HashSet};
use anyhow::Result;
#[cfg(feature = "cache")]
use deepbiop_utils::io::write_parquet_for_batches;
use log::warn;
use noodles::fasta;
use numpy::convert::ToPyArray;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;

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
#[pyclass(name = "RecordData", module = "deepbiop.fq")]
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
        // Assuming Python objects are tuples of (id, seq, qual)
        let (id, seq, qual): (String, String, String) = ob.extract()?;
        Ok(PyRecordData(encode::RecordData {
            id: id.into(),
            seq: seq.into(),
            qual: qual.into(),
        }))
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyRecordData {
    #[new]
    fn new(id: String, seq: String, qual: String) -> Self {
        Self(encode::RecordData {
            id: id.into(),
            seq: seq.into(),
            qual: qual.into(),
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

    #[getter]
    fn qual(&self) -> String {
        self.0.qual.to_string()
    }
    #[setter]
    fn set_qual(&mut self, qual: String) {
        self.0.qual = qual.into();
    }
}

#[gen_stub_pyfunction(module = "deepbiop.fq")]
#[pyfunction]
#[pyo3(signature = (records_data, file_path=None))]
fn write_fq(records_data: Vec<PyRecordData>, file_path: Option<PathBuf>) -> Result<()> {
    let records: Vec<encode::RecordData> = records_data
        .into_par_iter()
        .map(|py_record| py_record.0)
        .collect();
    io::write_fq(&records, file_path)
}

#[gen_stub_pyfunction(module = "deepbiop.fq")]
#[pyfunction]
fn write_fq_parallel(
    records_data: Vec<PyRecordData>,
    file_path: PathBuf,
    threads: usize,
) -> Result<()> {
    let records: Vec<encode::RecordData> = records_data
        .into_par_iter()
        .map(|py_record| py_record.0)
        .collect();

    io::write_bgzip_fq_parallel(&records, file_path, Some(threads))
}

#[cfg(feature = "cache")]
#[gen_stub_pyfunction(module = "deepbiop.fq")]
#[pyfunction]
fn encode_fq_path_to_parquet_chunk(
    fq_path: PathBuf,
    chunk_size: usize,
    parallel: bool,
    bases: String,
    qual_offset: usize,
) -> Result<()> {
    let option = encode::EncoderOptionBuilder::default()
        .bases(bases.as_bytes().to_vec())
        .qual_offset(qual_offset as u8)
        .build()?;

    let mut fq_encoder = encode::ParquetEncoderBuilder::default()
        .option(option)
        .build()?;
    fq_encoder.encode_chunk(&fq_path, chunk_size, parallel)?;
    Ok(())
}

#[gen_stub_pyfunction(module = "deepbiop.fq")]
#[pyfunction]
#[pyo3(signature = (fq_path, bases, qual_offset, result_path=None))]
fn encode_fq_path_to_parquet(
    fq_path: PathBuf,
    bases: String,
    qual_offset: usize,
    result_path: Option<PathBuf>,
) -> Result<()> {
    let option = encode::EncoderOptionBuilder::default()
        .bases(bases.as_bytes().to_vec())
        .qual_offset(qual_offset as u8)
        .build()?;

    let mut fq_encoder = encode::ParquetEncoderBuilder::default()
        .option(option)
        .build()?;
    let (record_batch, schema) = fq_encoder.encode(&fq_path)?;

    // result file is fq_path with .parquet extension
    let parquet_path = if let Some(path) = result_path {
        if path.with_extension("parquet").exists() {
            warn!("{} already exists, overwriting", path.display());
        }
        path.with_extension("parquet")
    } else {
        fq_path.with_extension("parquet")
    };

    #[cfg(feature = "cache")]
    {
        write_parquet_for_batches(parquet_path, &record_batch, schema)?;
        Ok(())
    }

    #[cfg(not(feature = "cache"))]
    {
        let _ = (parquet_path, record_batch, schema); // Silence unused warnings
        Err(anyhow::anyhow!("Parquet encoding requires 'cache' feature"))
    }
}

#[gen_stub_pyfunction(module = "deepbiop.fq")]
#[pyfunction]
fn encode_fq_paths_to_parquet(
    fq_path: Vec<PathBuf>,
    bases: String,
    qual_offset: usize,
) -> Result<()> {
    for path in &fq_path {
        encode_fq_path_to_parquet(path.clone(), bases.clone(), qual_offset, None)?;
    }
    Ok(())
}

#[gen_stub_pyfunction(module = "deepbiop.fq")]
#[pyfunction]
fn get_label_region(labels: Vec<i8>) -> Vec<(usize, usize)> {
    utils::get_label_region(&labels)
        .par_iter()
        .map(|r| (r.start, r.end))
        .collect()
}

#[gen_stub_pyfunction(module = "deepbiop.fq")]
#[pyfunction]
fn convert_multiple_fqs_to_one_fq(
    paths: Vec<PathBuf>,
    result_path: PathBuf,
    parallel: bool,
) -> Result<()> {
    if paths.is_empty() {
        return Ok(());
    }
    io::convert_multiple_fqs_to_one_bgzip_fq(&paths, result_path, parallel)?;
    Ok(())
}

/// Convert ASCII quality to Phred score for Phred+33 encoding
#[gen_stub_pyfunction(module = "deepbiop.fq")]
#[pyfunction]
pub fn encode_qual(qual: String, qual_offset: u8) -> Vec<u8> {
    let quals = qual.as_bytes();
    quals
        .par_iter()
        .map(|&q| {
            // Convert ASCII to Phred score for Phred+33 encoding
            q - qual_offset
        })
        .collect()
}

#[gen_stub_pyfunction(module = "deepbiop.fq")]
#[pyfunction]
pub fn test_predicts(predicts: Vec<PyRef<predicts::Predict>>) {
    predicts.iter().for_each(|predict| {
        println!("id: {}", predict.id);
        println!("seq: {}", predict.seq);
        println!("prediction: {:?}", predict.prediction);
        println!("is_truncated: {}", predict.is_truncated);
    });
}

#[gen_stub_pyfunction(module = "deepbiop.fq")]
#[pyfunction]
pub fn load_predicts_from_batch_pt(
    pt_path: PathBuf,
    ignore_label: i64,
    id_table: HashMap<i64, char>,
) -> Result<HashMap<String, Predict>> {
    predicts::load_predicts_from_batch_pt(pt_path, ignore_label, &id_table)
}

#[gen_stub_pyfunction(module = "deepbiop.fq")]
#[pyfunction]
#[pyo3(signature = (pt_path, ignore_label, id_table, max_predicts=None))]
pub fn load_predicts_from_batch_pts(
    pt_path: PathBuf,
    ignore_label: i64,
    id_table: HashMap<i64, char>,
    max_predicts: Option<usize>,
) -> Result<HashMap<String, Predict>> {
    predicts::load_predicts_from_batch_pts(pt_path, ignore_label, &id_table, max_predicts)
}

#[gen_stub_pyfunction(module = "deepbiop.fq")]
#[pyfunction]
pub fn fastq_to_fasta(fastq_path: PathBuf, fasta_path: PathBuf) -> Result<()> {
    let fa_records = io::fastq_to_fasta(&fastq_path)?;
    let handle = std::fs::File::create(fasta_path)?;
    let mut writer = fasta::io::Writer::new(handle);
    for record in fa_records {
        writer.write_record(&record)?;
    }
    Ok(())
}

#[gen_stub_pyfunction(module = "deepbiop.fq")]
#[pyfunction(name = "select_record_from_fq")]
pub fn py_select_record_from_fq(
    selected_reads: Vec<String>,
    fq: PathBuf,
    output: PathBuf,
) -> Result<()> {
    let selected_reads: HashSet<BString> =
        selected_reads.into_par_iter().map(|s| s.into()).collect();

    let records = io::select_record_from_fq(fq, &selected_reads)?;
    io::write_fq_for_noodle_record(&records, output)?;
    Ok(())
}

#[gen_stub_pyfunction(module = "deepbiop.fq")]
#[pyfunction(name = "select_record_from_fq_by_random")]
pub fn py_select_record_from_fq_by_random(
    fq: PathBuf,
    number: usize,
    output: PathBuf,
) -> Result<()> {
    let records = io::select_record_from_fq_by_random(fq, number)?;
    io::write_fq_for_noodle_record(&records, output)?;
    Ok(())
}

/// Streaming FASTQ dataset for efficient iteration over large files.
///
/// This dataset provides memory-efficient streaming iteration over FASTQ files,
/// reading records one at a time without loading the entire file into memory.
/// Yields records as dictionaries with NumPy arrays for zero-copy efficiency.
///
/// # Examples
///
/// ```python
/// from deepbiop.fq import FastqStreamDataset
///
/// dataset = FastqStreamDataset("data.fastq.gz")
/// for record in dataset:
///     seq = record['sequence']  # NumPy array
///     qual = record['quality']   # NumPy array
///     print(f"ID: {record['id']}, Length: {len(seq)}")
/// ```
#[gen_stub_pyclass]
#[pyclass(name = "FastqStreamDataset", module = "deepbiop.fq")]
pub struct PyFastqStreamDataset {
    file_path: String,
    size_hint: Option<usize>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyFastqStreamDataset {
    /// Create a new streaming FASTQ dataset.
    ///
    /// # Arguments
    ///
    /// * `file_path` - Path to FASTQ file (supports .fastq, .fastq.gz, .fastq.bgz)
    ///
    /// # Returns
    ///
    /// FastqStreamDataset instance
    ///
    /// # Raises
    ///
    /// * `FileNotFoundError` - If file doesn't exist
    /// * `IOError` - If file cannot be opened
    #[new]
    fn new(file_path: String) -> PyResult<Self> {
        let size_hint = Some(
            count_records_efficient(&file_path)
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?,
        );

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
    /// - 'quality': np.ndarray (uint8) - Quality score bytes
    /// - 'description': Optional[str] - Sequence description
    fn __iter__(slf: PyRef<'_, Self>) -> PyResult<Py<PyFastqStreamIterator>> {
        let iter = FastqStreamIterator::open(&slf.file_path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        Py::new(slf.py(), PyFastqStreamIterator { iter })
    }

    /// Get estimated number of records in dataset.
    ///
    /// # Returns
    ///
    /// int - Estimated record count (0 if unavailable)
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
    /// # Note
    ///
    /// Each call opens the file fresh and reads sequentially up to `index`,
    /// so random access is O(n) in the index. Iteration is preferred for
    /// sequential access.
    fn __getitem__(&self, index: usize, py: Python) -> PyResult<Py<PyDict>> {
        let mut iter = FastqStreamIterator::open(&self.file_path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

        match iter.nth(index) {
            None => Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "Index {} out of range for dataset with {} records",
                index,
                self.size_hint.unwrap_or(0)
            ))),
            Some(Ok(record)) => {
                let dict = PyDict::new(py);
                dict.set_item("id", record.id)?;
                dict.set_item("sequence", record.sequence.to_pyarray(py))?;

                if let Some(quality) = record.quality_scores {
                    dict.set_item("quality", quality.to_pyarray(py))?;
                } else {
                    dict.set_item("quality", py.None())?;
                }

                dict.set_item("description", record.description)?;
                Ok(dict.into())
            }
            Some(Err(e)) => Err(pyo3::exceptions::PyIOError::new_err(format!(
                "Failed to read record at index {}: {}",
                index, e
            ))),
        }
    }

    /// Human-readable representation.
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "FastqStreamDataset(file='{}', records={})",
            self.file_path,
            self.size_hint
                .map_or("unknown".to_string(), |n| n.to_string())
        ))
    }

    /// Provide arguments for __new__ during unpickling.
    ///
    /// This is called by pickle to get arguments to pass to __new__().
    ///
    /// # Returns
    ///
    /// Tuple of (file_path,) to pass to __new__
    fn __getnewargs__(&self) -> PyResult<(String,)> {
        Ok((self.file_path.clone(),))
    }

    /// Pickling support for multiprocessing (DataLoader with num_workers > 0).
    ///
    /// # Returns
    ///
    /// Dict with file_path and size_hint for reconstruction
    fn __getstate__(&self, py: Python) -> PyResult<Py<PyDict>> {
        let state = PyDict::new(py);
        state.set_item("file_path", &self.file_path)?;
        state.set_item("size_hint", self.size_hint)?;
        Ok(state.into())
    }

    /// Unpickling support for multiprocessing.
    ///
    /// # Arguments
    ///
    /// * `state` - Dict with file_path and size_hint from __getstate__
    fn __setstate__(&mut self, state: &Bound<'_, PyDict>) -> PyResult<()> {
        self.file_path = state.get_item("file_path")?.unwrap().extract()?;
        self.size_hint = state.get_item("size_hint")?.unwrap().extract()?;
        Ok(())
    }
}

/// Iterator for streaming FASTQ dataset.
#[gen_stub_pyclass]
#[pyclass(unsendable, name = "FastqStreamIterator", module = "deepbiop.fq")]
pub struct PyFastqStreamIterator {
    iter: FastqStreamIterator,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyFastqStreamIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python) -> PyResult<Option<Py<PyDict>>> {
        match self.iter.next() {
            None => Ok(None),
            Some(Ok(record)) => {
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
            Some(Err(e)) => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Failed to read FASTQ record: {}",
                e
            ))),
        }
    }
}

// register fq sub_module
pub fn register_fq_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let sub_module_name = "fq";
    let child_module = PyModule::new(parent_module.py(), sub_module_name)?;

    child_module.add_class::<PyRecordData>()?;
    child_module.add_class::<encode::EncoderOption>()?;
    child_module.add_class::<encode::ParquetEncoder>()?;
    child_module.add_class::<predicts::Predict>()?;

    // Add encoding classes
    child_module.add_class::<encode::onehot::python::PyOneHotEncoder>()?;
    child_module.add_class::<encode::integer::python::PyIntegerEncoder>()?;

    // Add filter classes
    child_module.add_class::<filter::python::PyLengthFilter>()?;
    child_module.add_class::<filter::python::PyQualityFilter>()?;
    child_module.add_class::<filter::python::PyDeduplicator>()?;
    child_module.add_class::<filter::python::PySubsampler>()?;

    // Add streaming dataset class
    child_module.add_class::<streaming::PyStreamingFastqDataset>()?;

    // Add augmentation classes
    crate::augment::python::register_augmentation_classes(&child_module)?;

    child_module.add_function(wrap_pyfunction!(py_select_record_from_fq, &child_module)?)?;
    child_module.add_function(wrap_pyfunction!(fastq_to_fasta, &child_module)?)?;
    child_module.add_function(wrap_pyfunction!(get_label_region, &child_module)?)?;
    child_module.add_function(wrap_pyfunction!(encode_qual, &child_module)?)?;
    child_module.add_function(wrap_pyfunction!(write_fq, &child_module)?)?;

    child_module.add_function(wrap_pyfunction!(utils::vertorize_target, &child_module)?)?;

    child_module.add_function(wrap_pyfunction!(write_fq, &child_module)?)?;
    child_module.add_function(wrap_pyfunction!(write_fq_parallel, &child_module)?)?;
    child_module.add_function(wrap_pyfunction!(encode_fq_path_to_parquet, &child_module)?)?;
    child_module.add_function(wrap_pyfunction!(encode_fq_paths_to_parquet, &child_module)?)?;
    #[cfg(feature = "cache")]
    child_module.add_function(wrap_pyfunction!(
        encode_fq_path_to_parquet_chunk,
        &child_module
    )?)?;

    child_module.add_function(wrap_pyfunction!(
        convert_multiple_fqs_to_one_fq,
        &child_module
    )?)?;

    child_module.add_function(wrap_pyfunction!(test_predicts, &child_module)?)?;
    child_module.add_function(wrap_pyfunction!(
        load_predicts_from_batch_pt,
        &child_module
    )?)?;
    child_module.add_function(wrap_pyfunction!(
        load_predicts_from_batch_pts,
        &child_module
    )?)?;

    child_module.add_function(wrap_pyfunction!(py_select_record_from_fq, &child_module)?)?;
    child_module.add_function(wrap_pyfunction!(
        py_select_record_from_fq_by_random,
        &child_module
    )?)?;

    child_module.add_class::<FastqDataset>()?;
    child_module.add_class::<FastqIterator>()?;
    child_module.add_class::<FastqRecord>()?;

    // Add streaming dataset classes
    child_module.add_class::<PyFastqStreamDataset>()?;
    child_module.add_class::<PyFastqStreamIterator>()?;

    parent_module.add_submodule(&child_module)?;

    Ok(())
}
