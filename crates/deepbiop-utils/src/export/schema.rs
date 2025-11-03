use arrow::datatypes::{DataType, Field, Schema};
use std::sync::Arc;

/// Standard schema for biological sequence data export.
///
/// This schema defines the structure for exporting FASTQ/FASTA records
/// to columnar formats like Parquet and Arrow, optimized for ML workflows.
///
/// # Columns
///
/// - `id`: String - Sequence identifier (header without '@' or '>')
/// - `sequence`: LargeBinary - Raw nucleotide/protein sequence as bytes
/// - `quality`: LargeBinary (optional) - Phred quality scores as bytes
/// - `length`: UInt32 - Sequence length for quick filtering
/// - `gc_content`: Float32 (optional) - GC content percentage (0-100)
///
/// # Example
///
/// ```rust
/// use deepbiop_utils::export::schema::create_sequence_schema;
///
/// let schema = create_sequence_schema(true, true);
/// assert_eq!(schema.fields().len(), 5);
/// ```
pub fn create_sequence_schema(include_quality: bool, include_gc: bool) -> Arc<Schema> {
    let mut fields = vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("sequence", DataType::LargeBinary, false),
        Field::new("length", DataType::UInt32, false),
    ];

    if include_quality {
        fields.push(Field::new("quality", DataType::LargeBinary, true));
    }

    if include_gc {
        fields.push(Field::new("gc_content", DataType::Float32, true));
    }

    Arc::new(Schema::new(fields))
}

/// Schema for encoded sequence data (one-hot or integer encoded).
///
/// This schema is designed for ML-ready sequence data where sequences
/// have been encoded as numerical arrays.
///
/// # Columns
///
/// - `id`: String - Sequence identifier
/// - `encoding`: List<UInt8> or List<Float32> - Encoded sequence representation
/// - `length`: UInt32 - Original sequence length
/// - `labels`: List<Int8> (optional) - Target labels for supervised learning
///
/// # Example
///
/// ```rust
/// use deepbiop_utils::export::schema::create_encoded_schema;
/// use arrow::datatypes::DataType;
///
/// let schema = create_encoded_schema(DataType::UInt8, true);
/// assert_eq!(schema.fields().len(), 4);
/// ```
pub fn create_encoded_schema(encoding_type: DataType, include_labels: bool) -> Arc<Schema> {
    let mut fields = vec![
        Field::new("id", DataType::Utf8, false),
        Field::new(
            "encoding",
            DataType::List(Arc::new(Field::new("item", encoding_type, false))),
            false,
        ),
        Field::new("length", DataType::UInt32, false),
    ];

    if include_labels {
        fields.push(Field::new(
            "labels",
            DataType::List(Arc::new(Field::new("item", DataType::Int8, false))),
            true,
        ));
    }

    Arc::new(Schema::new(fields))
}

/// Schema for BAM alignment data export.
///
/// Designed for exporting alignment information for ML analysis
/// of mapped reads.
///
/// # Columns
///
/// - `read_name`: String - Read identifier
/// - `reference_name`: String - Reference sequence name
/// - `position`: Int64 - Alignment position (0-based)
/// - `mapping_quality`: UInt8 - MAPQ score
/// - `cigar`: String - CIGAR string
/// - `sequence`: LargeBinary - Read sequence
/// - `quality`: LargeBinary - Base quality scores
/// - `flags`: UInt16 - SAM flags
pub fn create_alignment_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("read_name", DataType::Utf8, false),
        Field::new("reference_name", DataType::Utf8, true),
        Field::new("position", DataType::Int64, true),
        Field::new("mapping_quality", DataType::UInt8, true),
        Field::new("cigar", DataType::Utf8, true),
        Field::new("sequence", DataType::LargeBinary, false),
        Field::new("quality", DataType::LargeBinary, true),
        Field::new("flags", DataType::UInt16, false),
    ]))
}

/// Calculate GC content from a sequence.
///
/// # Arguments
///
/// * `sequence` - Byte slice containing nucleotide sequence
///
/// # Returns
///
/// GC content as percentage (0.0 - 100.0)
///
/// # Example
///
/// ```rust
/// use deepbiop_utils::export::schema::calculate_gc_content;
///
/// let seq = b"ATGCGCAT";
/// let gc = calculate_gc_content(seq);
/// assert!((gc - 50.0).abs() < 0.01);
/// ```
pub fn calculate_gc_content(sequence: &[u8]) -> f32 {
    if sequence.is_empty() {
        return 0.0;
    }

    let gc_count = sequence
        .iter()
        .filter(|&&b| b == b'G' || b == b'C' || b == b'g' || b == b'c')
        .count();

    (gc_count as f32 / sequence.len() as f32) * 100.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sequence_schema_minimal() {
        let schema = create_sequence_schema(false, false);
        assert_eq!(schema.fields().len(), 3);
        assert_eq!(schema.field(0).name(), "id");
        assert_eq!(schema.field(1).name(), "sequence");
        assert_eq!(schema.field(2).name(), "length");
    }

    #[test]
    fn test_sequence_schema_full() {
        let schema = create_sequence_schema(true, true);
        assert_eq!(schema.fields().len(), 5);
        assert_eq!(schema.field(3).name(), "quality");
        assert_eq!(schema.field(4).name(), "gc_content");
    }

    #[test]
    fn test_encoded_schema() {
        let schema = create_encoded_schema(DataType::UInt8, true);
        assert_eq!(schema.fields().len(), 4);
        assert_eq!(schema.field(1).name(), "encoding");
        assert_eq!(schema.field(3).name(), "labels");
    }

    #[test]
    fn test_alignment_schema() {
        let schema = create_alignment_schema();
        assert_eq!(schema.fields().len(), 8);
        assert_eq!(schema.field(0).name(), "read_name");
        assert_eq!(schema.field(7).name(), "flags");
    }

    #[test]
    fn test_gc_content() {
        assert_eq!(calculate_gc_content(b""), 0.0);
        assert!((calculate_gc_content(b"ATGC") - 50.0).abs() < 0.01);
        assert!((calculate_gc_content(b"GGGG") - 100.0).abs() < 0.01);
        assert!((calculate_gc_content(b"AAAA") - 0.0).abs() < 0.01);
        assert!((calculate_gc_content(b"atgc") - 50.0).abs() < 0.01); // lowercase
    }
}
