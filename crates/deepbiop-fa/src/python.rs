use std::path::PathBuf;

use crate::encode::Encoder;
use crate::{
    encode::{self},
    io,
};

use anyhow::Result;
use deepbiop_utils::io as deepbiop_io;
use log::warn;
use pyo3::prelude::*;
use rayon::prelude::*;

use pyo3_stub_gen::derive::*;

#[gen_stub_pymethods]
#[pymethods]
impl encode::ParquetEncoder {
    #[new]
    fn py_new(option: encode::FaEncoderOption) -> Self {
        encode::ParquetEncoder::new(option)
    }
}

#[gen_stub_pyclass]
#[pyclass(name = "RecordData", module = "deepbiop.fa")]
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

#[gen_stub_pyfunction(module = "deepbiop.fa")]
#[pyfunction]
fn write_fa(records_data: Vec<PyRecordData>, file_path: Option<PathBuf>) -> Result<()> {
    let records: Vec<encode::RecordData> = records_data
        .into_par_iter()
        .map(|py_record| py_record.0)
        .collect();
    io::write_fa(&records, file_path)
}

#[gen_stub_pyfunction(module = "deepbiop.fa")]
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
    io::write_zip_fa_parallel(&records, file_path, Some(threads))
}

#[gen_stub_pyfunction(module = "deepbiop.fa")]
#[pyfunction]
fn encode_fa_path_to_parquet_chunk(
    fa_path: PathBuf,
    chunk_size: usize,
    parallel: bool,
    bases: String,
    qual_offset: usize,
) -> Result<()> {
    let option = encode::FaEncoderOptionBuilder::default()
        .bases(bases.as_bytes().to_vec())
        .qual_offset(qual_offset as u8)
        .build()?;

    let mut fa_encoder = encode::ParquetEncoderBuilder::default()
        .option(option)
        .build()?;
    fa_encoder.encode_chunk(&fa_path, chunk_size, parallel)?;
    Ok(())
}

#[gen_stub_pyfunction(module = "deepbiop.fa")]
#[pyfunction]
fn encode_fa_path_to_parquet(
    fa_path: PathBuf,
    bases: String,
    qual_offset: usize,
    result_path: Option<PathBuf>,
) -> Result<()> {
    let option = encode::FaEncoderOptionBuilder::default()
        .bases(bases.as_bytes().to_vec())
        .qual_offset(qual_offset as u8)
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
    deepbiop_io::write_parquet(parquet_path, record_batch, schema)?;
    Ok(())
}

#[gen_stub_pyfunction(module = "deepbiop.fa")]
#[pyfunction]
fn encode_fa_paths_to_parquet(
    fa_path: Vec<PathBuf>,
    bases: String,
    qual_offset: usize,
) -> Result<()> {
    fa_path.iter().for_each(|path| {
        encode_fa_path_to_parquet(path.clone(), bases.clone(), qual_offset, None).unwrap();
    });
    Ok(())
}

#[gen_stub_pyfunction(module = "deepbiop.fa")]
#[pyfunction]
fn convert_multiple_fas_to_one_fa(
    paths: Vec<PathBuf>,
    result_path: PathBuf,
    parallel: bool,
) -> Result<()> {
    if paths.is_empty() {
        return Ok(());
    }

    let is_zip = paths[0].extension().unwrap() == "gz";

    if is_zip {
        io::convert_multiple_fas_to_one_zip_fa(&paths, result_path, parallel)?;
    } else {
        io::convert_multiple_zip_fas_to_one_zip_fa(&paths, result_path, parallel)?;
    }

    Ok(())
}

// register fq sub_module
pub fn register_fa_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let sub_module_name = "fa";
    let child_module = PyModule::new_bound(parent_module.py(), sub_module_name)?;

    child_module.add_class::<PyRecordData>()?;
    child_module.add_class::<encode::FaEncoderOption>()?;
    child_module.add_class::<encode::ParquetEncoder>()?;

    child_module.add_function(wrap_pyfunction!(write_fa, &child_module)?)?;
    child_module.add_function(wrap_pyfunction!(write_fa_parallel, &child_module)?)?;
    child_module.add_function(wrap_pyfunction!(encode_fa_path_to_parquet, &child_module)?)?;
    child_module.add_function(wrap_pyfunction!(encode_fa_paths_to_parquet, &child_module)?)?;
    child_module.add_function(wrap_pyfunction!(
        encode_fa_path_to_parquet_chunk,
        &child_module
    )?)?;

    child_module.add_function(wrap_pyfunction!(
        convert_multiple_fas_to_one_fa,
        &child_module
    )?)?;

    parent_module.add_submodule(&child_module)?;

    Ok(())
}
