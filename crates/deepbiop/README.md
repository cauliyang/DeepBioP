# deepbiop

[![crates.io](https://img.shields.io/crates/v/deepbiop.svg)](https://crates.io/crates/deepbiop)
[![docs.rs](https://docs.rs/deepbiop/badge.svg)](https://docs.rs/deepbiop)
[![license](https://img.shields.io/crates/l/deepbiop.svg)](https://github.com/cauliyang/DeepBioP/blob/main/LICENSE)

Deep Learning Processing Library for Biological Data - Rust Crate

This is the umbrella crate that re-exports functionality from specialized crates in the DeepBioP workspace. Use feature flags to include only the functionality you need.

## Features

The `deepbiop` crate uses feature flags to enable optional functionality. By default, only `deepbiop-core` is included.

### Available Features

| Feature | Description | Includes |
|---------|-------------|----------|
| `default` | Core functionality only | `deepbiop-core` |
| `fastq` | FASTQ/FASTA file processing | `deepbiop-fq` |
| `bam` | BAM/SAM alignment processing | `deepbiop-bam` |
| `fasta` | FASTA file processing | `deepbiop-fa` |
| `utils` | Utility functions and helpers | `deepbiop-utils` |
| `vcf` | VCF variant file processing | `deepbiop-vcf` |
| `gtf` | GTF/GFF annotation processing | `deepbiop-gtf` |

### Feature Combinations

You can combine multiple features as needed:

```toml
# Cargo.toml

# Only FASTQ processing
[dependencies]
deepbiop = { version = "0.2", features = ["fastq"] }

# FASTQ + BAM processing
[dependencies]
deepbiop = { version = "0.2", features = ["fastq", "bam"] }

# All biological file formats
[dependencies]
deepbiop = { version = "0.2", features = ["fastq", "bam", "fasta", "vcf", "gtf"] }

# Everything including utilities
[dependencies]
deepbiop = { version = "0.2", features = ["fastq", "bam", "fasta", "vcf", "gtf", "utils"] }
```

## Usage Examples

### FASTQ Sequence Encoding

```rust
use deepbiop::fq::{OneHotEncoder, IntegerEncoder};
use deepbiop::core::types::EncodingType;
use deepbiop::fq::encode::AmbiguousStrategy;

// One-hot encoding for CNNs/RNNs
let encoder = OneHotEncoder::new(EncodingType::DNA, AmbiguousStrategy::Skip);
let encoded = encoder.encode(b"ACGTACGT")?;
println!("Shape: {:?}", encoded.shape());  // [8, 4]

// Integer encoding for Transformers
let int_encoder = IntegerEncoder::new(EncodingType::DNA);
let int_encoded = int_encoder.encode(b"ACGTACGT")?;
println!("Encoded: {:?}", int_encoded);  // [0, 1, 2, 3, 0, 1, 2, 3]
```

### BAM Alignment Processing

```rust
use deepbiop::bam::{BamReader, AlignmentFeatures};
use std::path::Path;

// Open BAM with multithreaded decompression
let mut reader = BamReader::open(Path::new("alignments.bam"), Some(4))?;

// Extract ML features
let features = reader.extract_features()?;
for feat in features.iter().take(5) {
    println!("MAPQ: {}, Identity: {:.3}",
             feat.mapping_quality, feat.identity());
}
```

### Data Augmentation

```rust
use deepbiop::fq::augment::{ReverseComplement, Mutator};

// Reverse complement
let rc = ReverseComplement::new();
let rc_seq = rc.apply(b"ACGTACGT")?;

// Random mutations (2% rate)
let mutator = Mutator::builder()
    .mutation_rate(0.02)
    .seed(Some(42))
    .build();
let mutated = mutator.apply(b"ACGTACGT")?;
```

### VCF Variant Analysis

```rust
use deepbiop::vcf::reader::VcfReader;

let mut reader = VcfReader::open("variants.vcf")?;
let variants = reader.read_all()?;

for variant in variants {
    if variant.is_snp() && variant.passes_filter() {
        println!("{}:{} {} > {:?}",
                 variant.chromosome, variant.position,
                 variant.reference_allele, variant.alternate_alleles);
    }
}
```

### GTF Annotation Queries

```rust
use deepbiop::gtf::reader::GtfReader;

let mut reader = GtfReader::open("annotations.gtf")?;

// Build gene index for fast lookups
let gene_index = reader.build_gene_index()?;
println!("Indexed {} genes", gene_index.len());

// Filter by feature type
let exons = reader.filter_by_type("exon")?;
println!("Found {} exons", exons.len());
```

## Direct Crate Dependencies

If you prefer to depend on individual crates directly (for finer control or smaller dependency trees), you can use:

- [`deepbiop-core`](https://crates.io/crates/deepbiop-core) - Core types and traits
- [`deepbiop-fq`](https://crates.io/crates/deepbiop-fq) - FASTQ/FASTA processing
- [`deepbiop-bam`](https://crates.io/crates/deepbiop-bam) - BAM/SAM processing
- [`deepbiop-fa`](https://crates.io/crates/deepbiop-fa) - FASTA processing
- [`deepbiop-vcf`](https://crates.io/crates/deepbiop-vcf) - VCF processing
- [`deepbiop-gtf`](https://crates.io/crates/deepbiop-gtf) - GTF/GFF processing
- [`deepbiop-utils`](https://crates.io/crates/deepbiop-utils) - Utilities

## Python Bindings

For Python users, install the `deepbiop` package via pip:

```bash
pip install deepbiop
```

See the [Python package](https://pypi.org/project/deepbiop/) for Python-specific documentation.

## CLI Tool

Command-line interface available via `deepbiop-cli`:

```bash
cargo install deepbiop-cli
dbp --help
```

## Minimum Supported Rust Version (MSRV)

This crate requires Rust **1.90.0** or later.

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines.
