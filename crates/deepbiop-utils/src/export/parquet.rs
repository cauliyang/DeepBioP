use anyhow::Result;
use arrow::datatypes::Schema;
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_writer::ArrowWriter;
use parquet::file::properties::WriterProperties;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use super::arrow::{ArrowExporter, SequenceRecord};

/// Parquet writer for biological sequence data.
///
/// Provides a high-level interface for exporting sequence data to Parquet format,
/// which is optimized for analytics and ML workflows with efficient compression
/// and columnar storage.
///
/// # Example
///
/// ```rust,no_run
/// use deepbiop_utils::export::parquet::ParquetWriter;
/// use deepbiop_utils::export::arrow::SequenceRecord;
/// use std::path::Path;
///
/// let records = vec![
///     SequenceRecord::new("seq1".to_string(), b"ACGT".to_vec(), Some(b"IIII".to_vec())),
/// ];
///
/// let writer = ParquetWriter::for_fastq();
/// writer.write(Path::new("output.parquet"), &records).unwrap();
/// ```
pub struct ParquetWriter {
    arrow_exporter: ArrowExporter,
    writer_props: WriterProperties,
}

impl ParquetWriter {
    /// Create a new Parquet writer with custom settings.
    ///
    /// # Arguments
    ///
    /// * `include_quality` - Include quality scores column
    /// * `include_gc` - Include GC content column
    pub fn new(include_quality: bool, include_gc: bool) -> Self {
        let arrow_exporter = ArrowExporter::new(include_quality, include_gc);
        let writer_props = WriterProperties::builder()
            .set_compression(parquet::basic::Compression::SNAPPY)
            .set_max_row_group_size(100_000)
            .set_write_batch_size(1024)
            .build();

        Self {
            arrow_exporter,
            writer_props,
        }
    }

    /// Create a writer configured for FASTQ data (with quality scores).
    pub fn for_fastq() -> Self {
        Self::new(true, true)
    }

    /// Create a writer configured for FASTA data (without quality scores).
    pub fn for_fasta() -> Self {
        Self::new(false, true)
    }

    /// Get the Arrow schema used by this writer.
    pub fn schema(&self) -> Arc<Schema> {
        self.arrow_exporter.schema()
    }

    /// Write a batch of sequence records to a Parquet file.
    ///
    /// # Arguments
    ///
    /// * `path` - Output file path
    /// * `records` - Sequence records to write
    ///
    /// # Errors
    ///
    /// Returns error if file creation, Arrow conversion, or Parquet writing fails
    pub fn write<P: AsRef<Path>>(&self, path: P, records: &[SequenceRecord]) -> Result<()> {
        let record_batch = self.arrow_exporter.export(records)?;
        self.write_batch(path, &record_batch)
    }

    /// Write an Arrow RecordBatch to a Parquet file.
    ///
    /// Lower-level method for when you already have an Arrow RecordBatch.
    ///
    /// # Arguments
    ///
    /// * `path` - Output file path
    /// * `record_batch` - Arrow RecordBatch to write
    pub fn write_batch<P: AsRef<Path>>(&self, path: P, record_batch: &RecordBatch) -> Result<()> {
        let file = File::create(path.as_ref())?;
        let schema = record_batch.schema();
        let mut writer = ArrowWriter::try_new(file, schema, Some(self.writer_props.clone()))?;

        writer.write(record_batch)?;
        writer.close()?;

        Ok(())
    }

    /// Write multiple batches to a Parquet file.
    ///
    /// Useful for processing large datasets in chunks to manage memory usage.
    ///
    /// # Arguments
    ///
    /// * `path` - Output file path
    /// * `record_batches` - Slice of record batches to write
    pub fn write_batches<P: AsRef<Path>>(
        &self,
        path: P,
        record_batches: &[RecordBatch],
    ) -> Result<()> {
        if record_batches.is_empty() {
            return Err(anyhow::anyhow!("No record batches to write"));
        }

        let file = File::create(path.as_ref())?;
        let schema = record_batches[0].schema();
        let mut writer = ArrowWriter::try_new(file, schema, Some(self.writer_props.clone()))?;

        for batch in record_batches {
            writer.write(batch)?;
        }

        writer.close()?;
        Ok(())
    }

    /// Write sequence records in chunks to manage memory.
    ///
    /// Processes records in batches of `chunk_size` to avoid loading
    /// all data into memory at once.
    ///
    /// # Arguments
    ///
    /// * `path` - Output file path
    /// * `records` - All sequence records to write
    /// * `chunk_size` - Number of records per batch
    pub fn write_chunked<P: AsRef<Path>>(
        &self,
        path: P,
        records: &[SequenceRecord],
        chunk_size: usize,
    ) -> Result<()> {
        let file = File::create(path.as_ref())?;
        let schema = self.arrow_exporter.schema();
        let mut writer = ArrowWriter::try_new(file, schema, Some(self.writer_props.clone()))?;

        for chunk in records.chunks(chunk_size) {
            let batch = self.arrow_exporter.export(chunk)?;
            writer.write(&batch)?;
        }

        writer.close()?;
        Ok(())
    }
}

impl Default for ParquetWriter {
    fn default() -> Self {
        Self::for_fastq()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    fn create_test_records() -> Vec<SequenceRecord> {
        vec![
            SequenceRecord::new(
                "seq1".to_string(),
                b"ACGTACGT".to_vec(),
                Some(b"IIIIIIII".to_vec()),
            ),
            SequenceRecord::new(
                "seq2".to_string(),
                b"GGGGCCCC".to_vec(),
                Some(b"!!!!####".to_vec()),
            ),
            SequenceRecord::new("seq3".to_string(), b"AAAATTTT".to_vec(), None),
        ]
    }

    #[test]
    fn test_parquet_writer_fastq() -> Result<()> {
        let records = create_test_records();
        let writer = ParquetWriter::for_fastq();

        let temp_file = NamedTempFile::new()?;
        writer.write(temp_file.path(), &records)?;

        // Verify file was created and has content
        let metadata = std::fs::metadata(temp_file.path())?;
        assert!(metadata.len() > 0);

        Ok(())
    }

    #[test]
    fn test_parquet_writer_fasta() -> Result<()> {
        let records = create_test_records();
        let writer = ParquetWriter::for_fasta();

        let temp_file = NamedTempFile::new()?;
        writer.write(temp_file.path(), &records)?;

        let metadata = std::fs::metadata(temp_file.path())?;
        assert!(metadata.len() > 0);

        Ok(())
    }

    #[test]
    fn test_write_chunked() -> Result<()> {
        let records = create_test_records();
        let writer = ParquetWriter::for_fastq();

        let temp_file = NamedTempFile::new()?;
        writer.write_chunked(temp_file.path(), &records, 1)?;

        let metadata = std::fs::metadata(temp_file.path())?;
        assert!(metadata.len() > 0);

        Ok(())
    }

    #[test]
    fn test_write_batches() -> Result<()> {
        let records = create_test_records();
        let exporter = ArrowExporter::for_fastq();

        let batch1 = exporter.export(&records[0..1])?;
        let batch2 = exporter.export(&records[1..3])?;

        let writer = ParquetWriter::for_fastq();
        let temp_file = NamedTempFile::new()?;

        writer.write_batches(temp_file.path(), &[batch1, batch2])?;

        let metadata = std::fs::metadata(temp_file.path())?;
        assert!(metadata.len() > 0);

        Ok(())
    }

    #[test]
    fn test_schema_access() {
        let writer = ParquetWriter::for_fastq();
        let schema = writer.schema();

        assert_eq!(schema.fields().len(), 5); // id, sequence, length, quality, gc_content
    }
}
