//! Tests for compressed FASTQ file handling.
//!
//! These tests verify that the dataset can handle various compression formats
//! efficiently and correctly.

/// Test reading uncompressed FASTQ files.
///
/// Requirements:
/// - FR-004: Support plain text FASTQ files
/// - Should read records correctly without decompression overhead
#[test]
#[ignore = "Requires FastqDataset implementation (T020)"]
fn test_uncompressed_fastq() {
    // use deepbiop_fq::dataset::FastqDataset;

    // let test_file = PathBuf::from("tests/data/test.fastq");
    // let dataset = FastqDataset::new(test_file).unwrap();

    // let records: Vec<_> = dataset.iter().collect::<Result<Vec<_>, _>>().unwrap();

    // assert!(!records.is_empty());
    // // Verify records are valid
    // for record in &records {
    //     assert!(!record.id.is_empty());
    //     assert!(!record.sequence.is_empty());
    //     assert!(record.quality_scores.is_some());
    // }
}

/// Test reading gzip-compressed FASTQ files.
///
/// Requirements:
/// - FR-005: Support gzip compression (.gz)
/// - Should transparently decompress during iteration
#[test]
#[ignore = "Requires FastqDataset implementation (T020)"]
fn test_gzip_compressed_fastq() {
    // use deepbiop_fq::dataset::FastqDataset;

    // let test_file = PathBuf::from("tests/data/test.fastq.gz");
    // let dataset = FastqDataset::new(test_file).unwrap();

    // let records: Vec<_> = dataset.iter().collect::<Result<Vec<_>, _>>().unwrap();

    // assert!(!records.is_empty());
}

/// Test reading bgzip-compressed FASTQ files.
///
/// Requirements:
/// - FR-006: Support bgzip compression (block gzip)
/// - Should handle BGZF format correctly
#[test]
#[ignore = "Requires FastqDataset implementation (T020)"]
fn test_bgzip_compressed_fastq() {
    // use deepbiop_fq::dataset::FastqDataset;

    // let test_file = PathBuf::from("tests/data/test.fastqbgz.gz");
    // let dataset = FastqDataset::new(test_file).unwrap();

    // let records: Vec<_> = dataset.iter().collect::<Result<Vec<_>, _>>().unwrap();

    // assert!(!records.is_empty());
}

/// Test that all compression formats produce identical results.
///
/// Requirements:
/// - Compression should be transparent to users
/// - Same records should be returned regardless of compression
#[test]
#[ignore = "Requires FastqDataset implementation (T020)"]
fn test_compression_consistency() {
    // use deepbiop_fq::dataset::FastqDataset;

    // let files = vec![
    //     PathBuf::from("tests/data/test.fastq"),
    //     PathBuf::from("tests/data/test.fastq.gz"),
    //     PathBuf::from("tests/data/test.fastqbgz.gz"),
    // ];

    // let mut all_records = Vec::new();

    // for file in files {
    //     if !file.exists() {
    //         continue;
    //     }

    //     let dataset = FastqDataset::new(file).unwrap();
    //     let records: Vec<_> = dataset.iter().collect::<Result<Vec<_>, _>>().unwrap();
    //     all_records.push(records);
    // }

    // // All files should produce same number of records
    // if all_records.len() > 1 {
    //     let first_count = all_records[0].len();
    //     for records in &all_records {
    //         assert_eq!(records.len(), first_count);
    //     }

    //     // Verify records are identical (same IDs, sequences)
    //     for i in 0..first_count {
    //         let first_record = &all_records[0][i];
    //         for records in &all_records[1..] {
    //             assert_eq!(records[i].id, first_record.id);
    //             assert_eq!(records[i].sequence, first_record.sequence);
    //         }
    //     }
    // }
}

/// Test performance difference between compression formats.
///
/// Requirements:
/// - Decompression should be fast enough for streaming
/// - bgzip should be competitive with gzip
#[test]
#[ignore = "Requires FastqDataset implementation and benchmark data (T020)"]
fn test_compression_performance() {
    // use deepbiop_fq::dataset::FastqDataset;
    // use std::time::Instant;

    // let files = vec![
    //     ("uncompressed", PathBuf::from("tests/data/test.fastq")),
    //     ("gzip", PathBuf::from("tests/data/test.fastq.gz")),
    //     ("bgzip", PathBuf::from("tests/data/test.fastqbgz.gz")),
    // ];

    // for (name, file) in files {
    //     if !file.exists() {
    //         continue;
    //     }

    //     let dataset = FastqDataset::new(file).unwrap();

    //     let start = Instant::now();
    //     let count = dataset.iter().count();
    //     let elapsed = start.elapsed();

    //     let throughput = count as f64 / elapsed.as_secs_f64();

    //     println!("{}: {} records in {:?} ({:.0} records/sec)",
    //         name, count, elapsed, throughput);

    //     // All formats should achieve reasonable throughput
    //     assert!(throughput > 10_000.0, "Throughput too low for {}", name);
    // }
}

/// Test automatic compression detection.
///
/// Requirements:
/// - Dataset should automatically detect compression type
/// - No manual configuration required
#[test]
#[ignore = "Requires FastqDataset implementation (T020)"]
fn test_automatic_compression_detection() {
    // use deepbiop_fq::dataset::FastqDataset;

    // let test_files = vec![
    //     PathBuf::from("tests/data/test.fastq"),
    //     PathBuf::from("tests/data/test.fastq.gz"),
    //     PathBuf::from("tests/data/test.fastqbgz.gz"),
    // ];

    // for file in test_files {
    //     if !file.exists() {
    //         continue;
    //     }

    //     // Should not require specifying compression type
    //     let dataset = FastqDataset::new(file).unwrap();
    //     let count = dataset.iter().count();
    //     assert!(count > 0);
    // }
}

/// Test handling of corrupted compressed files.
///
/// Requirements:
/// - Should detect and report compression errors
/// - Should not panic on corrupted data
#[test]
#[ignore = "Requires FastqDataset implementation and corrupted test file (T020)"]
fn test_corrupted_compression() {
    // use deepbiop_fq::dataset::FastqDataset;

    // let test_file = PathBuf::from("tests/data/corrupted.fastq.gz");
    // if !test_file.exists() {
    //     return;
    // }

    // let dataset = FastqDataset::new(test_file).unwrap();

    // // Should get an error when iterating
    // let mut got_error = false;
    // for result in dataset.iter() {
    //     if result.is_err() {
    //         got_error = true;
    //         break;
    //     }
    // }

    // assert!(got_error, "Should detect compression corruption");
}

/// Test that size_hint works differently for compressed files.
///
/// Requirements:
/// - Uncompressed files can provide accurate size hints
/// - Compressed files may return None (unknown)
#[test]
#[ignore = "Requires FastqDataset implementation (T020)"]
fn test_size_hint_with_compression() {
    // use deepbiop_fq::dataset::FastqDataset;

    // // Uncompressed file
    // let uncompressed = PathBuf::from("tests/data/test.fastq");
    // if uncompressed.exists() {
    //     let dataset = FastqDataset::new(uncompressed).unwrap();
    //     let hint = dataset.size_hint();
    //     // Uncompressed files should provide hint
    //     assert!(hint.is_some());
    // }

    // // Compressed file
    // let compressed = PathBuf::from("tests/data/test.fastq.gz");
    // if compressed.exists() {
    //     let dataset = FastqDataset::new(compressed).unwrap();
    //     let hint = dataset.size_hint();
    //     // Compressed files may not provide hint
    //     // (implementation-dependent)
    // }
}
