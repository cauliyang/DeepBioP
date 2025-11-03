use crate::blat;
use crate::export;
use crate::io;

use crate::{
    blat::PslAlignment,
    export::arrow::SequenceRecord,
    interval::{self, GenomicInterval, Overlap},
    strategy,
};

use ahash::HashMap;
use anyhow::Result;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::ops::Range;
use std::path::PathBuf;

use pyo3_stub_gen::derive::*;

#[gen_stub_pymethods]
#[pymethods]
impl PslAlignment {
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "PslAlignment(qname={}, qsize={}, qstart={}, qend={}, qmatch={}, tname={}, tsize={}, tstart={}, tend={}, identity={})",
            self.qname, self.qsize, self.qstart, self.qend, self.qmatch, self.tname, self.tsize, self.tstart, self.tend, self.identity
        ))
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl GenomicInterval {
    #[new]
    fn py_new(chr: &str, start: usize, end: usize) -> Self {
        GenomicInterval {
            chr: chr.into(),
            start,
            end,
        }
    }

    #[getter]
    fn get_chr(&self) -> String {
        self.chr.to_string()
    }
    #[setter]
    fn set_chr(&mut self, chr: &str) {
        self.chr = chr.into();
    }

    #[pyo3(name = "overlap")]
    fn py_overlap(&self, other: &GenomicInterval) -> bool {
        self.overlap(other)
    }

    fn __repr__(&self) -> String {
        format!(
            "Segment(chr={}, start={}, end={})",
            self.chr, self.start, self.end
        )
    }
}

#[gen_stub_pyfunction(module = "deepbiop.utils")]
#[pyfunction]
fn majority_voting(labels: Vec<i8>, window_size: usize) -> Vec<i8> {
    strategy::majority_voting(&labels, window_size)
}

/// Parse PSL file by query name.
#[gen_stub_pyfunction(module = "deepbiop.utils")]
#[pyfunction]
fn parse_psl_by_qname(file_path: PathBuf) -> Result<HashMap<String, Vec<blat::PslAlignment>>> {
    blat::parse_psl_by_qname(file_path)
}

#[allow(clippy::type_complexity)]
#[gen_stub_pyfunction(module = "deepbiop.utils")]
#[pyfunction]
fn remove_intervals_and_keep_left(
    seq: String,
    intervals: Vec<(usize, usize)>,
) -> Result<(Vec<String>, Vec<(usize, usize)>)> {
    let intervals: Vec<Range<usize>> = intervals
        .par_iter()
        .map(|(start, end)| *start..*end)
        .collect();

    let (seqs, intevals) = interval::remove_intervals_and_keep_left(seq.as_bytes(), &intervals)?;
    Ok((
        seqs.par_iter().map(|s| s.to_string()).collect(),
        intevals.par_iter().map(|r| (r.start, r.end)).collect(),
    ))
}

#[gen_stub_pyfunction(module = "deepbiop.utils")]
#[pyfunction]
fn generate_unmaped_intervals(
    input: Vec<(usize, usize)>,
    total_length: usize,
) -> Vec<(usize, usize)> {
    let ranges: Vec<Range<usize>> = input.par_iter().map(|(start, end)| *start..*end).collect();
    interval::generate_unmaped_intervals(&ranges, total_length)
        .par_iter()
        .map(|r| (r.start, r.end))
        .collect()
}

/// Check the compression type of a file.
///
/// Args:
///     path: Path to the file to check
///
/// Returns:
///     The compression type of the file (None, Gzip, Bzip2, Xz)
///
/// Raises:
///     IOError: If the file cannot be opened or read
#[gen_stub_pyfunction(module = "deepbiop.utils")]
#[pyfunction(name = "check_compressed_type")]
fn py_check_compressed_type(path: PathBuf) -> Result<io::CompressedType> {
    io::check_compressed_type(path)
}

/// Export sequences to Parquet format.
///
/// Args:
///     path: Output Parquet file path
///     ids: Sequence identifiers
///     sequences: Sequence data (as bytes)
///     qualities: Optional quality scores (as bytes)
///
/// Example:
///     >>> export_to_parquet("output.parquet", ["seq1"], [b"ACGT"], [b"IIII"])
#[gen_stub_pyfunction(module = "deepbiop.utils")]
#[pyfunction]
#[pyo3(signature = (path, ids, sequences, qualities=None))]
fn export_to_parquet(
    path: PathBuf,
    ids: Vec<String>,
    sequences: Vec<Vec<u8>>,
    qualities: Option<Vec<Vec<u8>>>,
) -> Result<()> {
    // Convert to SequenceRecord
    let records: Vec<SequenceRecord> = ids
        .into_par_iter()
        .zip(sequences.into_par_iter())
        .enumerate()
        .map(|(i, (id, seq))| {
            let qual = qualities.as_ref().and_then(|quals| quals.get(i).cloned());
            SequenceRecord::new(id, seq, qual)
        })
        .collect();

    // Export based on whether we have quality scores
    let writer = if qualities.is_some() {
        export::parquet::ParquetWriter::for_fastq()
    } else {
        export::parquet::ParquetWriter::for_fasta()
    };

    writer.write(path, &records)
}

/// Export sequences to NumPy format (integer encoded).
///
/// Args:
///     path: Output .npy file path
///     sequences: Sequence data (as bytes)
///     alphabet: Encoding alphabet (default: b"ACGT")
///
/// Example:
///     >>> export_to_numpy_int("output.npy", [b"ACGT", b"GGCC"], b"ACGT")
#[gen_stub_pyfunction(module = "deepbiop.utils")]
#[pyfunction]
#[pyo3(signature = (path, sequences, alphabet=None))]
fn export_to_numpy_int(
    path: PathBuf,
    sequences: Vec<Vec<u8>>,
    alphabet: Option<Vec<u8>>,
) -> Result<()> {
    let alphabet = alphabet.unwrap_or_else(|| b"ACGT".to_vec());
    let exporter = export::numpy::NumpyExporter;
    exporter.export_integer(path, &sequences, &alphabet)
}

/// Export sequences to NumPy format (one-hot encoded).
///
/// Args:
///     path: Output .npy file path
///     sequences: Sequence data (as bytes)
///     alphabet: Encoding alphabet (default: b"ACGT")
///
/// Example:
///     >>> export_to_numpy_onehot("output.npy", [b"ACGT"], b"ACGT")
#[gen_stub_pyfunction(module = "deepbiop.utils")]
#[pyfunction]
#[pyo3(signature = (path, sequences, alphabet=None))]
fn export_to_numpy_onehot(
    path: PathBuf,
    sequences: Vec<Vec<u8>>,
    alphabet: Option<Vec<u8>>,
) -> Result<()> {
    let alphabet = alphabet.unwrap_or_else(|| b"ACGT".to_vec());
    let exporter = export::numpy::NumpyExporter;
    exporter.export_onehot(path, &sequences, &alphabet)
}

/// Export quality scores to NumPy format.
///
/// Args:
///     path: Output .npy file path
///     qualities: Quality score strings (as bytes)
///     offset: Phred quality offset (33 for Phred+33, 64 for Phred+64)
///
/// Example:
///     >>> export_quality_to_numpy("qual.npy", [b"IIII"], 33)
#[gen_stub_pyfunction(module = "deepbiop.utils")]
#[pyfunction]
#[pyo3(signature = (path, qualities, offset=33))]
fn export_quality_to_numpy(path: PathBuf, qualities: Vec<Vec<u8>>, offset: u8) -> Result<()> {
    let exporter = export::numpy::NumpyExporter;
    exporter.export_quality(path, &qualities, offset)
}

/// Export FASTQ data to paired NumPy files (sequences + qualities).
///
/// Args:
///     seq_path: Output .npy file for sequences
///     qual_path: Output .npy file for qualities
///     sequences: Sequence data (as bytes)
///     qualities: Quality scores (as bytes)
///     alphabet: Encoding alphabet (default: b"ACGT")
///     offset: Phred quality offset (default: 33)
///
/// Example:
///     >>> export_fastq_to_numpy("seq.npy", "qual.npy", [b"ACGT"], [b"IIII"])
#[gen_stub_pyfunction(module = "deepbiop.utils")]
#[pyfunction]
#[pyo3(signature = (seq_path, qual_path, sequences, qualities, alphabet=None, offset=33))]
fn export_fastq_to_numpy(
    seq_path: PathBuf,
    qual_path: PathBuf,
    sequences: Vec<Vec<u8>>,
    qualities: Vec<Vec<u8>>,
    alphabet: Option<Vec<u8>>,
    offset: u8,
) -> Result<()> {
    let alphabet = alphabet.unwrap_or_else(|| b"ACGT".to_vec());
    export::numpy::export_fastq_to_numpy(
        seq_path, qual_path, &sequences, &qualities, &alphabet, offset,
    )
}

// register utils module
pub fn register_utils_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let sub_module_name = "utils";
    let child_module = PyModule::new(parent_module.py(), sub_module_name)?;

    child_module.add_class::<GenomicInterval>()?;
    child_module.add_class::<PslAlignment>()?;
    child_module.add_class::<io::CompressedType>()?;

    child_module.add_function(wrap_pyfunction!(majority_voting, &child_module)?)?;
    child_module.add_function(wrap_pyfunction!(crate::highlight_targets, &child_module)?)?;
    child_module.add_function(wrap_pyfunction!(parse_psl_by_qname, &child_module)?)?;
    child_module.add_function(wrap_pyfunction!(
        remove_intervals_and_keep_left,
        &child_module
    )?)?;
    child_module.add_function(wrap_pyfunction!(generate_unmaped_intervals, &child_module)?)?;
    child_module.add_function(wrap_pyfunction!(py_check_compressed_type, &child_module)?)?;

    // Export utilities
    child_module.add_function(wrap_pyfunction!(export_to_parquet, &child_module)?)?;
    child_module.add_function(wrap_pyfunction!(export_to_numpy_int, &child_module)?)?;
    child_module.add_function(wrap_pyfunction!(export_to_numpy_onehot, &child_module)?)?;
    child_module.add_function(wrap_pyfunction!(export_quality_to_numpy, &child_module)?)?;
    child_module.add_function(wrap_pyfunction!(export_fastq_to_numpy, &child_module)?)?;

    parent_module.add_submodule(&child_module)?;
    Ok(())
}
