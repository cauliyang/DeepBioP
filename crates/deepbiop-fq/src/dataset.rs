#[cfg(feature = "python")]
use pyo3::types::{PyAny, PyDict, PyList};
use std::fs::File;
#[cfg(feature = "python")]
use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::Result;
use noodles::fastq;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use rayon::prelude::*;
use std::io::{BufReader, Read};

#[cfg(feature = "python")]
use pyo3_stub_gen::derive::*;

#[cfg_attr(feature = "python", gen_stub_pyclass)]
#[cfg_attr(
    feature = "python",
    pyclass(get_all, name = "FastqRecord", module = "deepbiop.fq")
)]
pub struct FastqRecord {
    pub header: String,
    pub sequence: String,
    pub quality: String,
}

#[cfg(feature = "python")]
#[gen_stub_pymethods]
#[cfg(feature = "python")]
#[pymethods]
impl FastqRecord {
    #[new]
    fn new(header: String, sequence: String, quality: String) -> Self {
        FastqRecord {
            header,
            sequence,
            quality,
        }
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "FastqRecord({}, seq_len={})",
            self.header,
            self.sequence.len()
        ))
    }
}

#[cfg_attr(feature = "python", gen_stub_pyclass)]
#[cfg_attr(
    feature = "python",
    pyclass(name = "FastqDataset", module = "deepbiop.fq")
)]
#[derive(Clone)]
#[allow(dead_code)]
pub struct FastqDataset {
    file_path: String,
    records_count: usize,
    chunk_size: usize,
    current_position: Arc<Mutex<usize>>,
}

#[cfg(feature = "python")]
#[gen_stub_pymethods]
#[cfg(feature = "python")]
#[pymethods]
impl FastqDataset {
    #[new]
    pub fn new(file_path: String, chunk_size: usize) -> Result<Self> {
        if !Path::new(&file_path).exists() {
            return Err(anyhow::anyhow!("File does not exist: {}", file_path));
        }

        let records_count = count_records_efficient(&file_path)?;
        Ok(Self {
            file_path,
            records_count,
            chunk_size,
            current_position: Arc::new(Mutex::new(0)),
        })
    }

    fn __len__(&self) -> PyResult<usize> {
        // Safer ceiling division that won't risk overflow or division by zero
        if self.chunk_size == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Chunk size cannot be zero",
            ));
        }
        Ok(self.records_count.div_ceil(self.chunk_size))
    }

    fn __getitem__(&self, idx: usize, py: Python) -> PyResult<Py<PyAny>> {
        if idx >= self.__len__()? {
            return Err(pyo3::exceptions::PyIndexError::new_err(
                "Index out of bounds",
            ));
        }

        let start_record = idx * self.chunk_size;
        let end_record = std::cmp::min(start_record + self.chunk_size, self.records_count);

        let records = self.get_records(start_record, end_record)?;
        let batch = PyList::new(
            py,
            records
                .into_iter()
                .map(|record| Py::new(py, record).unwrap()),
        )?;
        Ok(batch.into())
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyResult<Py<FastqIterator>> {
        // Reset the current position when starting a new iteration
        if let Ok(mut pos) = slf.current_position.lock() {
            *pos = 0;
        }

        Python::attach(|py| {
            let dataset_clone = Py::new(py, slf.clone())?;

            let iter = FastqIterator {
                dataset: dataset_clone,
                current_batch: 0,
            };

            Py::new(py, iter)
        })
    }

    #[staticmethod]
    fn from_file(file_path: &str, chunk_size: usize) -> PyResult<Self> {
        FastqDataset::new(file_path.to_string(), chunk_size)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
    }

    fn get_stats(&self, py: Python) -> PyResult<Py<PyAny>> {
        let dict = PyDict::new(py);
        dict.set_item("total_records", self.records_count)?;
        dict.set_item("batches", self.__len__()?)?;
        dict.set_item("chunk_size", self.chunk_size)?;
        dict.set_item("file_path", &self.file_path)?;
        Ok(dict.into())
    }

    pub fn get_records(&self, start: usize, end: usize) -> PyResult<Vec<FastqRecord>> {
        // Validate input parameters to prevent bugs
        if end < start {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "End index cannot be less than start index",
            ));
        }

        // For large skips, use indexed access if possible
        if start > 1000 {
            return self.get_records_indexed(start, end);
        }

        let file = File::open(&self.file_path).map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!("Failed to open file: {}", e))
        })?;

        let mut reader = fastq::io::Reader::new(BufReader::with_capacity(65536, file)); // Increase buffer size
        let mut records = Vec::with_capacity(end - start); // Pre-allocate vector
        let mut current_pos = 0;

        for result in reader.records() {
            let record =
                result.map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

            // Skip records before start
            if current_pos < start {
                current_pos += 1;
                continue;
            }

            // Stop if we've reached the end
            if current_pos >= end {
                break;
            }

            // Convert noodles record to our FastqRecord
            let header = String::from_utf8_lossy(record.name()).to_string();
            let sequence = String::from_utf8_lossy(record.sequence()).to_string();
            let quality = String::from_utf8_lossy(record.quality_scores()).to_string();

            records.push(FastqRecord {
                header,
                sequence,
                quality,
            });

            current_pos += 1;
        }

        Ok(records)
    }

    // Optimized indexed access implementation
    fn get_records_indexed(&self, start: usize, end: usize) -> PyResult<Vec<FastqRecord>> {
        // Use parallel processing for large batches with better chunking strategy
        if end - start > 1000 {
            // Calculate optimal chunk size based on number of threads and work size
            let thread_count = rayon::current_num_threads();
            let mut chunk_size = (end - start) / thread_count;

            // Ensure chunks aren't too small (processing overhead becomes significant)
            const MIN_CHUNK_SIZE: usize = 250;
            if chunk_size < MIN_CHUNK_SIZE && (end - start) > MIN_CHUNK_SIZE {
                chunk_size = MIN_CHUNK_SIZE;
            }

            if chunk_size > 0 {
                // Divide work into chunks
                let chunks: Vec<_> = (start..end)
                    .step_by(chunk_size)
                    .map(|chunk_start| {
                        let chunk_end = std::cmp::min(chunk_start + chunk_size, end);
                        (chunk_start, chunk_end)
                    })
                    .collect();

                let results: Result<Vec<_>, _> = chunks
                    .par_iter()
                    .map(|(chunk_start, chunk_end)| {
                        self.get_records_sequential(*chunk_start, *chunk_end)
                    })
                    .collect();

                // Use more efficient collection approach
                return match results {
                    Ok(chunk_records) => {
                        // Pre-calculate total capacity needed
                        let total_capacity: usize = chunk_records.iter().map(|v| v.len()).sum();
                        let mut records = Vec::with_capacity(total_capacity);

                        for chunk in chunk_records {
                            records.extend(chunk);
                        }

                        Ok(records)
                    }
                    Err(e) => Err(e),
                };
            }
        }

        // Fall back to sequential reading
        self.get_records_sequential(start, end)
    }

    // More efficient sequential record reading with better error handling
    fn get_records_sequential(&self, start: usize, end: usize) -> PyResult<Vec<FastqRecord>> {
        // Add cache to avoid reopening the file multiple times for adjacent calls
        let file = File::open(&self.file_path).map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!("Failed to open file: {}", e))
        })?;

        let mut reader = fastq::io::Reader::new(BufReader::with_capacity(65536, file));
        let mut records = Vec::with_capacity(end - start); // Pre-allocate vector

        // Use enumerate with bounds checking
        for (current_pos, result) in reader.records().enumerate() {
            if current_pos >= end {
                break;
            }

            // Skip processing if before start index
            if current_pos < start {
                // Still check for errors in skipped records
                result.map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "Error reading record at position {}: {}",
                        current_pos, e
                    ))
                })?;
                continue;
            }

            // Process record
            match result {
                Ok(record) => {
                    // Convert directly to String to avoid multiple allocations
                    records.push(FastqRecord {
                        header: String::from_utf8_lossy(record.name()).into_owned(),
                        sequence: String::from_utf8_lossy(record.sequence()).into_owned(),
                        quality: String::from_utf8_lossy(record.quality_scores()).into_owned(),
                    });
                }
                Err(e) => {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "Error reading record at position {}: {}",
                        current_pos, e
                    )));
                }
            }
        }

        Ok(records)
    }
}

#[cfg_attr(feature = "python", gen_stub_pyclass)]
#[cfg_attr(
    feature = "python",
    pyclass(name = "FastqIterator", module = "deepbiop.fq")
)]
#[allow(dead_code)]
pub struct FastqIterator {
    #[cfg(feature = "python")]
    dataset: Py<FastqDataset>,
    current_batch: usize,
}

#[cfg(feature = "python")]
#[gen_stub_pymethods]
#[cfg(feature = "python")]
#[pymethods]
impl FastqIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> PyResult<Option<Py<PyAny>>> {
        Python::attach(|py| {
            // Use direct field access for better performance
            let current_batch = slf.current_batch;

            // First check if we've reached the end to avoid unnecessary work
            let total_batches;
            let batch_result;
            {
                let dataset = slf.dataset.borrow(py);
                total_batches = match dataset.__len__() {
                    Ok(len) => len,
                    Err(e) => return Err(e),
                };

                if current_batch >= total_batches {
                    return Ok(None);
                }

                // Get the batch
                batch_result = match dataset.__getitem__(current_batch, py) {
                    Ok(batch) => Ok(Some(batch)),
                    Err(e) => Err(e),
                };
            } // dataset borrow is dropped here

            // Increment counter after retrieval, now that dataset borrow is dropped
            if batch_result.is_ok() {
                slf.current_batch += 1;
            }

            batch_result
        })
    }
}

// Non-Python methods for FastqDataset
impl FastqDataset {
    /// Get total number of records in the dataset.
    pub fn records_count(&self) -> usize {
        self.records_count
    }
}

// IterableDataset trait implementation for streaming access
impl deepbiop_core::dataset::IterableDataset for FastqDataset {
    fn iter(
        &self,
    ) -> Box<
        dyn Iterator<
                Item = deepbiop_core::dataset::DatasetResult<deepbiop_core::seq::SequenceRecord>,
            > + '_,
    > {
        Box::new(FastqStreamIterator::new(&self.file_path))
    }

    fn paths(&self) -> Vec<std::path::PathBuf> {
        vec![std::path::PathBuf::from(&self.file_path)]
    }

    fn size_hint(&self) -> Option<usize> {
        Some(self.records_count)
    }
}

/// Streaming iterator for IterableDataset trait implementation.
///
/// This provides true streaming access without batching.
struct FastqStreamIterator {
    reader: fastq::io::Reader<BufReader<Box<dyn Read>>>,
}

impl FastqStreamIterator {
    fn new(file_path: &str) -> Self {
        // Create reader with compression support
        let file_reader = deepbiop_utils::io::create_reader_for_compressed_file(file_path)
            .expect("Failed to create file reader");

        let buffered = BufReader::new(file_reader);
        let reader = fastq::io::Reader::new(buffered);

        Self { reader }
    }
}

impl Iterator for FastqStreamIterator {
    type Item = deepbiop_core::dataset::DatasetResult<deepbiop_core::seq::SequenceRecord>;

    fn next(&mut self) -> Option<Self::Item> {
        // Read next record
        match self.reader.records().next() {
            None => None, // EOF
            Some(Ok(record)) => {
                // Convert to SequenceRecord
                let id = String::from_utf8_lossy(record.name()).into_owned();
                let sequence = record.sequence().to_vec();
                let quality_scores = Some(record.quality_scores().to_vec());
                let desc_bstr = record.definition().description();
                let description = if desc_bstr.is_empty() {
                    None
                } else {
                    Some(desc_bstr.to_string())
                };

                let seq_record = deepbiop_core::seq::SequenceRecord::new(
                    id,
                    sequence,
                    quality_scores,
                    description,
                );

                Some(Ok(seq_record))
            }
            Some(Err(e)) => {
                // Return error
                Some(Err(deepbiop_core::error::DPError::InvalidValue(format!(
                    "Failed to read FASTQ record: {}",
                    e
                ))))
            }
        }
    }
}

// More robust and efficient record counting
#[allow(dead_code)]
fn count_records_efficient(file_path: &str) -> Result<usize> {
    let file = File::open(file_path)?;
    let file_size = file.metadata()?.len() as usize;

    // If file is empty, return 0 immediately
    if file_size == 0 {
        return Ok(0);
    }

    // If file is small, count directly
    if file_size < 10_000_000 {
        // 10MB threshold
        return count_records_exact(file_path);
    }

    // Take larger sample for better accuracy on large files
    let sample_size = std::cmp::min(file_size / 10, 5_000_000); // 5MB max, or 10% of file
    let mut reader = BufReader::with_capacity(65536, file);
    let mut sample_data = Vec::with_capacity(sample_size);

    reader
        .by_ref()
        .take(sample_size as u64)
        .read_to_end(&mut sample_data)?;

    // Count lines in sample more efficiently
    let line_count = sample_data.iter().filter(|&&b| b == b'\n').count();

    // Handle edge case of no newlines in sample
    if line_count == 0 {
        // Fall back to exact counting since estimation is unreliable
        return count_records_exact(file_path);
    }

    // Improve estimation by checking if sample starts/ends with complete records
    let avg_lines_per_byte = line_count as f64 / sample_data.len() as f64;
    let estimated_total_lines = (avg_lines_per_byte * file_size as f64).ceil() as usize;

    // FASTQ records are always 4 lines
    let estimated_records = estimated_total_lines / 4;

    Ok(estimated_records)
}

// Original counting function renamed for exact counting
#[allow(dead_code)]
fn count_records_exact(file_path: &str) -> Result<usize> {
    let file = File::open(file_path)?;
    let mut reader = fastq::io::Reader::new(BufReader::with_capacity(65536, file));
    let count = reader.records().count();
    Ok(count)
}

// Legacy function to maintain compatibility
#[cfg(feature = "python")]
#[allow(dead_code)]
fn count_records(file_path: &str) -> PyResult<usize> {
    count_records_efficient(file_path)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
}
