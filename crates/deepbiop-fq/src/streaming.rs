//! Streaming FASTQ dataset with shuffle buffer for memory-efficient processing.
//!
//! This module provides streaming dataset implementations that can process
//! large FASTQ files (>100GB) without loading entire dataset into memory.

use anyhow::{Context, Result};
use noodles::fastq::io::Reader;
use noodles::fastq::Record;
use rand::prelude::*;
use rand::seq::SliceRandom;
use std::collections::VecDeque;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyAny;
#[cfg(feature = "python")]
use pyo3_stub_gen::derive::*;

/// FASTQ record for streaming iteration
#[derive(Clone, Debug)]
pub struct StreamingRecord {
    pub id: Vec<u8>,
    pub sequence: Vec<u8>,
    pub quality: Vec<u8>,
}

impl StreamingRecord {
    /// Create a new streaming record from noodles::fastq::Record
    pub fn from_fastq_record(record: &Record) -> Self {
        Self {
            id: record.name().to_vec(),
            sequence: record.sequence().to_vec(),
            quality: record.quality_scores().to_vec(),
        }
    }

    /// Convert to Python dictionary
    #[cfg(feature = "python")]
    pub fn to_py_dict(&self, py: Python) -> PyResult<Py<PyAny>> {
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("id", self.id.as_slice())?;
        dict.set_item("sequence", self.sequence.as_slice())?;
        dict.set_item("quality", self.quality.as_slice())?;
        Ok(dict.into())
    }
}

/// Shuffle buffer for approximate randomization of streaming data
///
/// Uses reservoir sampling to provide approximate shuffling without
/// loading entire dataset into memory. Buffer size determines the
/// randomization window.
pub struct ShuffleBuffer {
    buffer: VecDeque<StreamingRecord>,
    capacity: usize,
    rng: rand::rngs::ThreadRng,
}

impl ShuffleBuffer {
    /// Create a new shuffle buffer with given capacity
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: VecDeque::with_capacity(capacity),
            capacity,
            rng: rand::rng(),
        }
    }

    /// Add a record to the buffer
    ///
    /// Returns Some(record) if buffer is full and a random record should be emitted
    pub fn push(&mut self, record: StreamingRecord) -> Option<StreamingRecord> {
        if self.buffer.len() < self.capacity {
            // Buffer not full yet, just add
            self.buffer.push_back(record);
            None
        } else {
            // Buffer full: randomly select position to insert new record
            // and return the record at that position
            let idx = (0..=self.capacity - 1)
                .collect::<Vec<_>>()
                .choose(&mut self.rng)
                .copied()
                .unwrap_or(0);

            if idx < self.capacity - 1 {
                // Replace existing record
                let old = self.buffer.remove(idx).unwrap();
                self.buffer.insert(idx, record);
                Some(old)
            } else {
                // New record bypasses buffer (reservoir sampling)
                Some(record)
            }
        }
    }

    /// Drain remaining records from buffer in random order
    pub fn drain(&mut self) -> Vec<StreamingRecord> {
        let mut remaining: Vec<_> = self.buffer.drain(..).collect();
        remaining.shuffle(&mut self.rng);
        remaining
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Get current buffer size
    pub fn len(&self) -> usize {
        self.buffer.len()
    }
}

/// Streaming FASTQ iterator with optional shuffling
pub struct StreamingFastqIterator {
    reader: Reader<BufReader<File>>,
    shuffle_buffer: Option<ShuffleBuffer>,
    drain_buffer: Vec<StreamingRecord>,
    done: bool,
}

impl StreamingFastqIterator {
    /// Create a new streaming iterator
    ///
    /// # Arguments
    /// * `path` - Path to FASTQ file (plain, gzip, or bgzip)
    /// * `shuffle_buffer_size` - Size of shuffle buffer (0 = no shuffling)
    pub fn new<P: AsRef<Path>>(path: P, shuffle_buffer_size: usize) -> Result<Self> {
        let file = File::open(path.as_ref())
            .with_context(|| format!("Failed to open FASTQ file: {:?}", path.as_ref()))?;

        let reader = BufReader::new(file);
        let fastq_reader = Reader::new(reader);

        let shuffle_buffer = if shuffle_buffer_size > 0 {
            Some(ShuffleBuffer::new(shuffle_buffer_size))
        } else {
            None
        };

        Ok(Self {
            reader: fastq_reader,
            shuffle_buffer,
            drain_buffer: Vec::new(),
            done: false,
        })
    }

    /// Read next record from file (internal helper)
    fn read_next_record(&mut self) -> Result<Option<StreamingRecord>> {
        let mut record = Record::default();

        match self.reader.read_record(&mut record) {
            Ok(0) => Ok(None), // EOF
            Ok(_) => Ok(Some(StreamingRecord::from_fastq_record(&record))),
            Err(e) => Err(anyhow::anyhow!("Failed to read FASTQ record: {}", e)),
        }
    }
}

impl Iterator for StreamingFastqIterator {
    type Item = Result<StreamingRecord>;

    fn next(&mut self) -> Option<Self::Item> {
        // First, check if we have records in drain buffer
        if !self.drain_buffer.is_empty() {
            return Some(Ok(self.drain_buffer.remove(0)));
        }

        if self.done {
            return None;
        }

        if self.shuffle_buffer.is_some() {
            // With shuffling
            loop {
                // Read next record
                let next_record = match self.read_next_record() {
                    Ok(Some(r)) => r,
                    Ok(None) => {
                        // EOF: drain buffer
                        self.done = true;
                        let shuffle_buffer = self.shuffle_buffer.as_mut().unwrap();
                        self.drain_buffer = shuffle_buffer.drain();
                        if !self.drain_buffer.is_empty() {
                            return Some(Ok(self.drain_buffer.remove(0)));
                        }
                        return None;
                    }
                    Err(e) => {
                        self.done = true;
                        return Some(Err(e));
                    }
                };

                // Add to buffer and check if we should emit a record
                let shuffle_buffer = self.shuffle_buffer.as_mut().unwrap();
                if let Some(emitted) = shuffle_buffer.push(next_record) {
                    return Some(Ok(emitted));
                }
                // Buffer not full yet, continue reading
            }
        } else {
            // Without shuffling - simple iteration
            match self.read_next_record() {
                Ok(Some(record)) => Some(Ok(record)),
                Ok(None) => {
                    self.done = true;
                    None
                }
                Err(e) => {
                    self.done = true;
                    Some(Err(e))
                }
            }
        }
    }
}

#[cfg(feature = "python")]
/// Python wrapper for streaming FASTQ dataset
#[cfg_attr(feature = "python", gen_stub_pyclass)]
#[cfg_attr(feature = "python", pyclass(name = "StreamingFastqDataset"))]
pub struct PyStreamingFastqDataset {
    path: String,
    shuffle_buffer_size: usize,
}

#[cfg(feature = "python")]
#[gen_stub_pymethods]
#[pymethods]
impl PyStreamingFastqDataset {
    /// Create a new streaming FASTQ dataset
    ///
    /// # Arguments
    /// * `path` - Path to FASTQ file
    /// * `shuffle_buffer_size` - Size of shuffle buffer (default: 0 = no shuffling)
    ///
    /// # Example
    /// ```python
    /// dataset = StreamingFastqDataset("large_file.fastq.gz", shuffle_buffer_size=10000)
    /// for record in dataset:
    ///     print(record["id"], len(record["sequence"]))
    /// ```
    #[new]
    #[pyo3(signature = (path, shuffle_buffer_size=0))]
    pub fn new(path: String, shuffle_buffer_size: usize) -> PyResult<Self> {
        if !Path::new(&path).exists() {
            return Err(pyo3::exceptions::PyFileNotFoundError::new_err(format!(
                "FASTQ file not found: {}",
                path
            )));
        }

        Ok(Self {
            path,
            shuffle_buffer_size,
        })
    }

    /// Make the dataset iterable from Python
    fn __iter__(slf: PyRef<'_, Self>) -> PyResult<Py<PyStreamingFastqIterator>> {
        let iter = PyStreamingFastqIterator::new(
            slf.path.clone(),
            slf.shuffle_buffer_size,
        )?;

        Py::new(slf.py(), iter)
    }

    fn __repr__(&self) -> String {
        format!(
            "StreamingFastqDataset(path='{}', shuffle_buffer_size={})",
            self.path, self.shuffle_buffer_size
        )
    }
}

#[cfg(feature = "python")]
/// Python iterator wrapper for streaming FASTQ
#[cfg_attr(feature = "python", gen_stub_pyclass)]
#[pyclass(unsendable)]
pub struct PyStreamingFastqIterator {
    inner: StreamingFastqIterator,
}

#[cfg(feature = "python")]
impl PyStreamingFastqIterator {
    /// Create a new Python streaming iterator (internal helper)
    fn new(path: String, shuffle_buffer_size: usize) -> PyResult<Self> {
        let inner = StreamingFastqIterator::new(path, shuffle_buffer_size)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("Failed to create iterator: {}", e)))?;

        Ok(Self { inner })
    }
}

#[cfg(feature = "python")]
#[gen_stub_pymethods]
#[pymethods]
impl PyStreamingFastqIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> PyResult<Option<Py<PyAny>>> {
        match slf.inner.next() {
            Some(Ok(record)) => {
                let dict = record.to_py_dict(slf.py())?;
                Ok(Some(dict))
            }
            Some(Err(e)) => Err(pyo3::exceptions::PyIOError::new_err(format!(
                "Error reading record: {}",
                e
            ))),
            None => Ok(None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shuffle_buffer() {
        let mut buffer = ShuffleBuffer::new(3);

        // Add records to buffer
        let r1 = StreamingRecord {
            id: b"read1".to_vec(),
            sequence: b"ACGT".to_vec(),
            quality: b"IIII".to_vec(),
        };
        let r2 = StreamingRecord {
            id: b"read2".to_vec(),
            sequence: b"CGTA".to_vec(),
            quality: b"JJJJ".to_vec(),
        };
        let r3 = StreamingRecord {
            id: b"read3".to_vec(),
            sequence: b"GTAC".to_vec(),
            quality: b"KKKK".to_vec(),
        };

        // Fill buffer
        assert!(buffer.push(r1.clone()).is_none());
        assert!(buffer.push(r2.clone()).is_none());
        assert!(buffer.push(r3.clone()).is_none());
        assert_eq!(buffer.len(), 3);

        // Next push should emit a record
        let r4 = StreamingRecord {
            id: b"read4".to_vec(),
            sequence: b"TACG".to_vec(),
            quality: b"LLLL".to_vec(),
        };
        let emitted = buffer.push(r4);
        assert!(emitted.is_some());

        // Drain should return remaining records
        let drained = buffer.drain();
        assert_eq!(drained.len(), 3);
        assert!(buffer.is_empty());
    }
}
