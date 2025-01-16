use bstr::BString;
use std::path::PathBuf;

use crate::{
    encode::{self, Encoder},
    io,
    predicts::{self, Predict},
    utils,
};

use ahash::{HashMap, HashSet};
use anyhow::Result;
use deepbiop_utils::io::write_parquet;
use log::warn;
use noodles::fasta;
use pyo3::prelude::*;
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
impl<'py> FromPyObject<'py> for PyRecordData {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
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
    write_parquet(parquet_path, record_batch, schema)?;
    Ok(())
}

#[gen_stub_pyfunction(module = "deepbiop.fq")]
#[pyfunction]
fn encode_fq_paths_to_parquet(
    fq_path: Vec<PathBuf>,
    bases: String,
    qual_offset: usize,
) -> Result<()> {
    fq_path.iter().for_each(|path| {
        encode_fq_path_to_parquet(path.clone(), bases.clone(), qual_offset, None).unwrap();
    });
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

    let is_zip = paths[0].extension().unwrap() == "gz";

    if is_zip {
        io::convert_multiple_fqs_to_one_bgzip_fq(&paths, result_path, parallel)?;
    } else {
        io::convert_multiple_bgzip_fqs_to_one_bgzip_fq(&paths, result_path, parallel)?;
    }

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
    let mut writer = fasta::Writer::new(handle);
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

// register fq sub_module
pub fn register_fq_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let sub_module_name = "fq";
    let child_module = PyModule::new(parent_module.py(), sub_module_name)?;

    child_module.add_class::<PyRecordData>()?;
    child_module.add_class::<encode::EncoderOption>()?;
    child_module.add_class::<encode::ParquetEncoder>()?;
    child_module.add_class::<predicts::Predict>()?;

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

    parent_module.add_submodule(&child_module)?;

    Ok(())
}
