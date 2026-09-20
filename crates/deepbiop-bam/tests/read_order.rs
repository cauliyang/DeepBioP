//! Regression test: reading a BAM must return records in file order.
//!
//! The record decode is serial, so the per-record work is parallelised with
//! `par_iter` (order-preserving). A `par_bridge().collect()` here returned
//! records in arbitrary order, which surfaced as intermittent test failures
//! once the default worker count became "all cores".

use deepbiop_bam::reader::BamReader;
use std::path::Path;

const TEST_BAM: &str = "tests/data/test_chimric_reads.bam";

fn names(threads: Option<usize>) -> Vec<String> {
    let mut reader = BamReader::open(Path::new(TEST_BAM), threads).unwrap();
    reader
        .read_all()
        .unwrap()
        .iter()
        .map(|record| String::from_utf8_lossy(record.name().unwrap_or_default()).to_string())
        .collect()
}

#[test]
fn read_all_preserves_file_order_across_worker_counts() {
    let single = names(Some(1));
    assert!(single.len() > 10, "fixture should contain records");

    assert_eq!(
        single,
        names(Some(2)),
        "worker_count=2 changed record order"
    );
    assert_eq!(
        single,
        names(None),
        "default worker count changed record order"
    );
}

#[test]
fn filter_and_feature_extraction_preserve_order() {
    // Order-preservation must hold for the derived readers too.
    let mut reader = BamReader::open(Path::new(TEST_BAM), None).unwrap();
    let all: Vec<_> = reader.read_all().unwrap();

    // Fixture MAPQ is mixed (3×1, 1×6, 1×33, 95×60), so this threshold keeps and
    // drops records: a vacuous filter would not test the ordering.
    const MIN_MAPQ: u8 = 20;
    let mut filtered_reader = BamReader::open(Path::new(TEST_BAM), None).unwrap();
    let filtered = filtered_reader.filter_by_mapping_quality(MIN_MAPQ).unwrap();

    let name_of = |r: &noodles::bam::Record| {
        String::from_utf8_lossy(r.name().unwrap_or_default()).to_string()
    };
    let expected: Vec<String> = all
        .iter()
        .filter(|r| {
            r.mapping_quality()
                .is_some_and(|mq| u8::from(mq) >= MIN_MAPQ)
        })
        .map(name_of)
        .collect();
    let actual: Vec<String> = filtered.iter().map(name_of).collect();

    assert!(
        !expected.is_empty() && expected.len() < all.len(),
        "fixture must exercise both kept and dropped records (kept {})",
        expected.len()
    );
    assert_eq!(
        actual, expected,
        "filter_by_mapping_quality reordered records"
    );

    let mut feature_reader = BamReader::open(Path::new(TEST_BAM), None).unwrap();
    let features = feature_reader.extract_features().unwrap();
    assert_eq!(
        features.len(),
        all.len(),
        "extract_features dropped records"
    );
}
