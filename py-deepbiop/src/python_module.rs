use std::path::PathBuf;

use ahash::HashMap;

use deepbiop_bam as bam;

use deepbiop_fq::{self as fastq, encode::Encoder, types::Element};

use anyhow::Result;
use log::warn;
use needletail::Sequence;
use numpy::{IntoPyArray, PyArray2, PyArray3};
use pyo3::prelude::*;
use rayon::prelude::*;

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

#[pyclass(name = "RecordData")]
pub struct PyRecordData(fastq::encode::RecordData);

impl From<fastq::encode::RecordData> for PyRecordData {
    fn from(data: fastq::encode::RecordData) -> Self {
        Self(data)
    }
}

// Implement FromPyObject for PyRecordData
impl<'py> FromPyObject<'py> for PyRecordData {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        // Assuming Python objects are tuples of (id, seq, qual)
        let (id, seq, qual): (String, String, String) = ob.extract()?;
        Ok(PyRecordData(fastq::encode::RecordData {
            id: id.into(),
            seq: seq.into(),
            qual: qual.into(),
        }))
    }
}

#[pymethods]
impl PyRecordData {
    #[new]
    fn new(id: String, seq: String, qual: String) -> Self {
        Self(fastq::encode::RecordData {
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

#[pyfunction]
fn write_fq(records_data: Vec<PyRecordData>, file_path: Option<PathBuf>) -> Result<()> {
    let records: Vec<fastq::encode::RecordData> = records_data
        .into_par_iter()
        .map(|py_record| py_record.0)
        .collect();
    fastq::io::write_fq(&records, file_path)
}

#[pyfunction]
fn write_fq_parallel(
    records_data: Vec<PyRecordData>,
    file_path: PathBuf,
    threads: usize,
) -> Result<()> {
    let records: Vec<fastq::encode::RecordData> = records_data
        .into_par_iter()
        .map(|py_record| py_record.0)
        .collect();

    fastq::io::write_zip_fq_parallel(&records, file_path, Some(threads))
}

#[pyfunction]
fn seq_to_kmers(seq: String, k: usize, overlap: bool) -> Vec<String> {
    let normalized_seq = seq.as_bytes().normalize(false);
    fastq::kmer::seq_to_kmers(&normalized_seq, k, overlap)
        .par_iter()
        .map(|s| String::from_utf8_lossy(s).to_string())
        .collect()
}

#[pyfunction]
fn kmers_to_seq(kmers: Vec<String>) -> Result<String> {
    let kmers_as_bytes: Vec<&[u8]> = kmers.par_iter().map(|s| s.as_bytes()).collect();
    Ok(String::from_utf8_lossy(&fastq::kmer::kmers_to_seq(kmers_as_bytes)?).to_string())
}

#[pyfunction]
fn generate_kmers_table(base: String, k: usize) -> fastq::types::Kmer2IdTable {
    let base = base.as_bytes();
    fastq::kmer::generate_kmers_table(base, k as u8)
}

#[pyfunction]
fn generate_kmers(base: String, k: usize) -> Vec<String> {
    let base = base.as_bytes();
    fastq::kmer::generate_kmers(base, k as u8)
        .into_iter()
        .map(|s| String::from_utf8_lossy(&s).to_string())
        .collect()
}

#[pyfunction]
fn normalize_seq(seq: String, iupac: bool) -> String {
    String::from_utf8_lossy(&seq.as_bytes().normalize(iupac)).to_string()
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
#[pyfunction]
fn encode_fq_paths_to_tensor(
    py: Python,
    fq_paths: Vec<PathBuf>,
    k: usize,
    bases: String,
    qual_offset: usize,
    vectorized_target: bool,
    parallel_for_files: bool,
    max_width: Option<usize>,
    max_seq_len: Option<usize>,
) -> Result<(
    Bound<'_, PyArray3<Element>>,
    Bound<'_, PyArray3<Element>>,
    Bound<'_, PyArray2<Element>>,
    HashMap<String, Element>,
)> {
    let option = fastq::encode::FqEncoderOptionBuilder::default()
        .kmer_size(k as u8)
        .bases(bases.as_bytes().to_vec())
        .qual_offset(qual_offset as u8)
        .vectorized_target(vectorized_target)
        .build()?;

    let mut fq_encoder = fastq::encode::TensorEncoderBuilder::default()
        .option(option)
        .tensor_max_width(max_width.unwrap_or(0))
        .tensor_max_seq_len(max_seq_len.unwrap_or(0))
        .build()?;

    let ((input, target), qual) = fq_encoder.encode_multiple(&fq_paths, parallel_for_files)?;

    let kmer2id: HashMap<String, fastq::types::Element> = fq_encoder
        .kmer2id_table
        .par_iter()
        .map(|(k, v)| (String::from_utf8_lossy(k).to_string(), *v))
        .collect();

    Ok((
        input.into_pyarray_bound(py),
        target.into_pyarray_bound(py),
        qual.into_pyarray_bound(py),
        kmer2id,
    ))
}
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
#[pyfunction]
fn encode_fq_path_to_tensor(
    py: Python,
    fq_path: PathBuf,
    k: usize,
    bases: String,
    qual_offset: usize,
    vectorized_target: bool,
    max_width: Option<usize>,
    max_seq_len: Option<usize>,
) -> Result<(
    Bound<'_, PyArray3<Element>>,
    Bound<'_, PyArray3<Element>>,
    Bound<'_, PyArray2<Element>>,
    HashMap<String, Element>,
)> {
    let option = fastq::encode::FqEncoderOptionBuilder::default()
        .kmer_size(k as u8)
        .bases(bases.as_bytes().to_vec())
        .qual_offset(qual_offset as u8)
        .vectorized_target(vectorized_target)
        .build()?;

    let mut fq_encoder = fastq::encode::TensorEncoderBuilder::default()
        .option(option)
        .tensor_max_width(max_width.unwrap_or(0))
        .tensor_max_seq_len(max_seq_len.unwrap_or(0))
        .build()?;

    let ((input, target), qual) = fq_encoder.encode(fq_path)?;

    let kmer2id: HashMap<String, Element> = fq_encoder
        .kmer2id_table
        .par_iter()
        .map(|(k, v)| (String::from_utf8_lossy(k).to_string(), *v))
        .collect();

    Ok((
        input.into_pyarray_bound(py),
        target.into_pyarray_bound(py),
        qual.into_pyarray_bound(py),
        kmer2id,
    ))
}

#[pyfunction]
fn encode_fq_path_to_json(
    fq_path: PathBuf,
    k: usize,
    bases: String,
    qual_offset: usize,
    vectorized_target: bool,
    result_path: Option<PathBuf>,
) -> Result<()> {
    let option = fastq::encode::FqEncoderOptionBuilder::default()
        .kmer_size(k as u8)
        .bases(bases.as_bytes().to_vec())
        .qual_offset(qual_offset as u8)
        .vectorized_target(vectorized_target)
        .build()?;

    let mut fq_encoder = fastq::encode::JsonEncoderBuilder::default()
        .option(option)
        .build()?;

    let result = fq_encoder.encode(&fq_path)?;

    // result file is fq_path with .parquet extension
    let json_path = if let Some(path) = result_path {
        if path.with_extension("json").exists() {
            warn!("{} already exists, overwriting", path.display());
        }
        path.with_extension("json")
    } else {
        fq_path.with_extension("json")
    };
    fastq::io::write_json(json_path, result)?;
    Ok(())
}

#[pyfunction]
fn encode_fq_path_to_parquet_chunk(
    fq_path: PathBuf,
    chunk_size: usize,
    parallel: bool,
    bases: String,
    qual_offset: usize,
    vectorized_target: bool,
) -> Result<()> {
    let option = fastq::encode::FqEncoderOptionBuilder::default()
        .kmer_size(0)
        .bases(bases.as_bytes().to_vec())
        .qual_offset(qual_offset as u8)
        .vectorized_target(vectorized_target)
        .build()?;

    let mut fq_encoder = fastq::encode::ParquetEncoderBuilder::default()
        .option(option)
        .build()?;
    fq_encoder.encode_chunk(&fq_path, chunk_size, parallel)?;
    Ok(())
}

#[pyfunction]
fn encode_fq_path_to_parquet(
    fq_path: PathBuf,
    bases: String,
    qual_offset: usize,
    vectorized_target: bool,
    result_path: Option<PathBuf>,
) -> Result<()> {
    let option = fastq::encode::FqEncoderOptionBuilder::default()
        .kmer_size(0)
        .bases(bases.as_bytes().to_vec())
        .qual_offset(qual_offset as u8)
        .vectorized_target(vectorized_target)
        .build()?;

    let mut fq_encoder = fastq::encode::ParquetEncoderBuilder::default()
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
    fastq::io::write_parquet(parquet_path, record_batch, schema)?;
    Ok(())
}

#[pyfunction]
fn encode_fq_paths_to_parquet(
    fq_path: Vec<PathBuf>,
    bases: String,
    qual_offset: usize,
    vectorized_target: bool,
) -> Result<()> {
    fq_path.iter().for_each(|path| {
        encode_fq_path_to_parquet(
            path.clone(),
            bases.clone(),
            qual_offset,
            vectorized_target,
            None,
        )
        .unwrap();
    });
    Ok(())
}

#[pyfunction]
fn get_label_region(labels: Vec<i8>) -> Vec<(usize, usize)> {
    fastq::utils::get_label_region(&labels)
        .par_iter()
        .map(|r| (r.start, r.end))
        .collect()
}

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
        fastq::io::convert_multiple_fqs_to_one_zip_fq(&paths, result_path, parallel)?;
    } else {
        fastq::io::convert_multiple_zip_fqs_to_one_zip_fq(&paths, result_path, parallel)?;
    }

    Ok(())
}

#[pyfunction]
fn reverse_complement(seq: String) -> String {
    String::from_utf8(seq.as_bytes().reverse_complement()).unwrap()
}

// register default sub_module
pub fn register_default_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let child_module = PyModule::new_bound(parent_module.py(), "default")?;
    child_module.add("QUAL_OFFSET", deepbiop_fq::default::QUAL_OFFSET)?;
    child_module.add(
        "BASES",
        String::from_utf8_lossy(deepbiop_fq::default::BASES),
    )?;
    child_module.add("KMER_SIZE", deepbiop_fq::default::KMER_SIZE)?;
    child_module.add("VECTORIZED_TARGET", deepbiop_fq::default::VECTORIZED_TARGET)?;
    parent_module.add_submodule(&child_module)?;
    Ok(())
}

// register fq sub_module
pub fn register_fq_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let child_module = PyModule::new_bound(parent_module.py(), "fq")?;

    child_module.add_class::<PyRecordData>()?;
    child_module.add_class::<fastq::encode::FqEncoderOption>()?;
    child_module.add_class::<fastq::encode::TensorEncoder>()?;
    child_module.add_class::<fastq::encode::JsonEncoder>()?;
    child_module.add_class::<fastq::encode::ParquetEncoder>()?;

    child_module.add_function(wrap_pyfunction!(get_label_region, &child_module)?)?;
    child_module.add_function(wrap_pyfunction!(encode_qual, &child_module)?)?;
    child_module.add_function(wrap_pyfunction!(write_fq, &child_module)?)?;

    child_module.add_function(wrap_pyfunction!(
        deepbiop_fq::kmer::vertorize_target,
        &child_module
    )?)?;
    child_module.add_function(wrap_pyfunction!(normalize_seq, &child_module)?)?;

    child_module.add_function(wrap_pyfunction!(seq_to_kmers, &child_module)?)?;
    child_module.add_function(wrap_pyfunction!(kmers_to_seq, &child_module)?)?;
    child_module.add_function(wrap_pyfunction!(generate_kmers, &child_module)?)?;
    child_module.add_function(wrap_pyfunction!(generate_kmers_table, &child_module)?)?;
    child_module.add_function(wrap_pyfunction!(encode_fq_paths_to_tensor, &child_module)?)?;

    child_module.add_function(wrap_pyfunction!(write_fq, &child_module)?)?;
    child_module.add_function(wrap_pyfunction!(write_fq_parallel, &child_module)?)?;
    child_module.add_function(wrap_pyfunction!(encode_fq_paths_to_tensor, &child_module)?)?;
    child_module.add_function(wrap_pyfunction!(encode_fq_path_to_tensor, &child_module)?)?;
    child_module.add_function(wrap_pyfunction!(encode_fq_path_to_parquet, &child_module)?)?;
    child_module.add_function(wrap_pyfunction!(encode_fq_paths_to_parquet, &child_module)?)?;
    child_module.add_function(wrap_pyfunction!(
        encode_fq_path_to_parquet_chunk,
        &child_module
    )?)?;
    child_module.add_function(wrap_pyfunction!(encode_fq_path_to_json, &child_module)?)?;

    child_module.add_function(wrap_pyfunction!(reverse_complement, &child_module)?)?;
    child_module.add_function(wrap_pyfunction!(
        convert_multiple_fqs_to_one_fq,
        &child_module
    )?)?;

    parent_module.add_submodule(&child_module)?;
    Ok(())
}

// register bam sub module
pub fn register_bam_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let child_module = PyModule::new_bound(parent_module.py(), "bam")?;
    child_module.add_function(wrap_pyfunction!(bam::left_right_soft_clip, &child_module)?)?;

    parent_module.add_submodule(&child_module)?;
    Ok(())
}
