use anyhow::Result;
use arrow::array::{
    ArrayRef, Float32Array, LargeBinaryArray, RecordBatch, StringArray, UInt32Array,
};
use arrow::datatypes::Schema;
use std::sync::Arc;

use super::schema::{calculate_gc_content, create_sequence_schema};

/// Represents a biological sequence record for Arrow export.
#[derive(Debug, Clone)]
pub struct SequenceRecord {
    pub id: String,
    pub sequence: Vec<u8>,
    pub quality: Option<Vec<u8>>,
}

impl SequenceRecord {
    /// Create a new sequence record.
    pub fn new(id: String, sequence: Vec<u8>, quality: Option<Vec<u8>>) -> Self {
        Self {
            id,
            sequence,
            quality,
        }
    }

    /// Get the sequence length.
    pub fn len(&self) -> u32 {
        self.sequence.len() as u32
    }

    /// Check if the sequence is empty.
    pub fn is_empty(&self) -> bool {
        self.sequence.is_empty()
    }

    /// Calculate GC content for this sequence.
    pub fn gc_content(&self) -> f32 {
        calculate_gc_content(&self.sequence)
    }
}

/// Arrow exporter for biological sequence data.
///
/// Converts biological sequences into Apache Arrow RecordBatch format
/// for efficient columnar processing and zero-copy sharing with other
/// tools like Polars, DuckDB, and pandas.
///
/// # Example
///
/// ```rust
/// use deepbiop_utils::export::arrow::{ArrowExporter, SequenceRecord};
///
/// let records = vec![
///     SequenceRecord::new("seq1".to_string(), b"ACGT".to_vec(), Some(b"IIII".to_vec())),
///     SequenceRecord::new("seq2".to_string(), b"GGCC".to_vec(), Some(b"!!!!".to_vec())),
/// ];
///
/// let exporter = ArrowExporter::new(true, true);
/// let batch = exporter.export(&records).unwrap();
/// assert_eq!(batch.num_rows(), 2);
/// ```
pub struct ArrowExporter {
    include_quality: bool,
    include_gc: bool,
    schema: Arc<Schema>,
}

impl ArrowExporter {
    /// Create a new Arrow exporter.
    ///
    /// # Arguments
    ///
    /// * `include_quality` - Include quality scores column
    /// * `include_gc` - Include GC content column
    pub fn new(include_quality: bool, include_gc: bool) -> Self {
        let schema = create_sequence_schema(include_quality, include_gc);
        Self {
            include_quality,
            include_gc,
            schema,
        }
    }

    /// Create an exporter for FASTQ data (includes quality).
    pub fn for_fastq() -> Self {
        Self::new(true, true)
    }

    /// Create an exporter for FASTA data (no quality).
    pub fn for_fasta() -> Self {
        Self::new(false, true)
    }

    /// Get the schema for this exporter.
    pub fn schema(&self) -> Arc<Schema> {
        Arc::clone(&self.schema)
    }

    /// Export a batch of sequence records to Arrow RecordBatch.
    ///
    /// # Arguments
    ///
    /// * `records` - Slice of SequenceRecord to export
    ///
    /// # Returns
    ///
    /// Arrow RecordBatch containing the sequence data
    ///
    /// # Errors
    ///
    /// Returns error if array construction or batch creation fails
    pub fn export(&self, records: &[SequenceRecord]) -> Result<RecordBatch> {
        // Build ID array
        let ids: StringArray = records.iter().map(|r| Some(r.id.as_str())).collect();

        // Build sequence array
        let sequences: LargeBinaryArray = records
            .iter()
            .map(|r| Some(r.sequence.as_slice()))
            .collect();

        // Build length array
        let lengths: UInt32Array = records.iter().map(|r| Some(r.len())).collect();

        // Build columns vec
        let mut columns: Vec<ArrayRef> =
            vec![Arc::new(ids), Arc::new(sequences), Arc::new(lengths)];

        // Add quality if needed
        if self.include_quality {
            let qualities: LargeBinaryArray =
                records.iter().map(|r| r.quality.as_deref()).collect();
            columns.push(Arc::new(qualities));
        }

        // Add GC content if needed
        if self.include_gc {
            let gc_contents: Float32Array = records.iter().map(|r| Some(r.gc_content())).collect();
            columns.push(Arc::new(gc_contents));
        }

        // Create record batch
        let batch = RecordBatch::try_new(Arc::clone(&self.schema), columns)?;

        Ok(batch)
    }

    /// Export multiple batches of records.
    ///
    /// Useful for processing large datasets in chunks.
    ///
    /// # Arguments
    ///
    /// * `record_batches` - Slice of record batches to export
    ///
    /// # Returns
    ///
    /// Vector of Arrow RecordBatches
    pub fn export_batches(
        &self,
        record_batches: &[Vec<SequenceRecord>],
    ) -> Result<Vec<RecordBatch>> {
        record_batches
            .iter()
            .map(|batch| self.export(batch))
            .collect()
    }
}

impl Default for ArrowExporter {
    fn default() -> Self {
        Self::new(true, true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn test_sequence_record() {
        let record =
            SequenceRecord::new("test".to_string(), b"ACGT".to_vec(), Some(b"IIII".to_vec()));

        assert_eq!(record.len(), 4);
        assert!(!record.is_empty());
        assert!((record.gc_content() - 50.0).abs() < 0.01);
    }

    #[test]
    fn test_arrow_exporter_fastq() -> Result<()> {
        let records = create_test_records();
        let exporter = ArrowExporter::for_fastq();

        let batch = exporter.export(&records)?;

        assert_eq!(batch.num_rows(), 3);
        assert_eq!(batch.num_columns(), 5); // id, sequence, length, quality, gc_content

        Ok(())
    }

    #[test]
    fn test_arrow_exporter_fasta() -> Result<()> {
        let records = create_test_records();
        let exporter = ArrowExporter::for_fasta();

        let batch = exporter.export(&records)?;

        assert_eq!(batch.num_rows(), 3);
        assert_eq!(batch.num_columns(), 4); // id, sequence, length, gc_content

        Ok(())
    }

    #[test]
    fn test_arrow_exporter_minimal() -> Result<()> {
        let records = create_test_records();
        let exporter = ArrowExporter::new(false, false);

        let batch = exporter.export(&records)?;

        assert_eq!(batch.num_rows(), 3);
        assert_eq!(batch.num_columns(), 3); // id, sequence, length

        Ok(())
    }

    #[test]
    fn test_export_batches() -> Result<()> {
        let batch1 = vec![create_test_records()[0].clone()];
        let batch2 = vec![
            create_test_records()[1].clone(),
            create_test_records()[2].clone(),
        ];

        let exporter = ArrowExporter::for_fastq();
        let batches = exporter.export_batches(&[batch1, batch2])?;

        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].num_rows(), 1);
        assert_eq!(batches[1].num_rows(), 2);

        Ok(())
    }

    #[test]
    fn test_empty_records() -> Result<()> {
        let records: Vec<SequenceRecord> = vec![];
        let exporter = ArrowExporter::for_fastq();

        let batch = exporter.export(&records)?;

        assert_eq!(batch.num_rows(), 0);
        assert_eq!(batch.num_columns(), 5);

        Ok(())
    }
}
