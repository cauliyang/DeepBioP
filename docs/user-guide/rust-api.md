# Rust API Usage Guide

This guide provides comprehensive examples for using DeepBioP's Rust library for high-performance biological data processing.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Feature Flags](#feature-flags)
4. [FASTQ Processing](#fastq-processing)
5. [FASTA Processing](#fasta-processing)
6. [BAM/SAM Processing](#bamsam-processing)
7. [VCF/GTF Processing](#vcfgtf-processing)
8. [Encoding for ML](#encoding-for-ml)
9. [Data Augmentation](#data-augmentation)
10. [Advanced Usage](#advanced-usage)

## Installation

Add DeepBioP to your `Cargo.toml`:

```toml
[dependencies]
deepbiop = { version = "0.1", features = ["fastq", "bam", "fasta"] }

# Or specific sub-crates
deepbiop-fq = "0.1"
deepbiop-bam = "0.1"
deepbiop-fa = "0.1"
deepbiop-core = "0.1"
```

### Minimum Supported Rust Version (MSRV)

DeepBioP requires Rust 1.90.0 or later.

## Quick Start

### Read FASTQ File

```rust
use deepbiop_fq::reader::FastqReader;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create reader
    let reader = FastqReader::from_path("data.fastq")?;

    // Iterate over records
    for record in reader {
        let record = record?;
        println!("ID: {}", record.id());
        println!("Sequence length: {}", record.sequence().len());
    }

    Ok(())
}
```

### Encode DNA Sequences

```rust
use deepbiop_fq::encoder::{OneHotEncoder, EncodingType, UnknownStrategy};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create encoder
    let encoder = OneHotEncoder::new(
        EncodingType::DNA,
        UnknownStrategy::Skip
    );

    // Encode sequence
    let sequence = b"ACGTACGT";
    let encoded = encoder.encode(sequence)?;

    println!("Shape: {:?}", encoded.shape());  // [8, 4]
    println!("Encoded: {:?}", encoded);

    Ok(())
}
```

## Feature Flags

DeepBioP uses feature flags to minimize compilation times:

```toml
[dependencies.deepbiop]
version = "0.1"
features = [
    "fastq",  # FASTQ file support
    "fasta",  # FASTA file support
    "bam",    # BAM/SAM support
    "utils",  # Utility functions
]
```

### Available Features

- `fastq`: FASTQ reading/writing, encoding, augmentation
- `fasta`: FASTA reading/writing
- `bam`: BAM/SAM reading, chimeric counting, BAM→FASTQ conversion
- `utils`: Compression utilities, parallel processing
- `default`: Includes `dep:deepbiop-core`

## FASTQ Processing

### Reading FASTQ Files

```rust
use deepbiop_fq::reader::FastqReader;

// Basic reading
let reader = FastqReader::from_path("input.fastq")?;
for result in reader {
    let record = result?;
    println!("{}: {}bp", record.id(), record.sequence().len());
}

// With gzip compression
let reader = FastqReader::from_path("input.fastq.gz")?;  // Auto-detects

// Read into vector
let reader = FastqReader::from_path("input.fastq")?;
let records: Vec<_> = reader.collect::<Result<Vec<_>, _>>()?;
println!("Total records: {}", records.len());
```

### Writing FASTQ Files

```rust
use deepbiop_fq::writer::FastqWriter;
use deepbiop_fq::record::FastqRecord;

// Create writer
let mut writer = FastqWriter::from_path("output.fastq")?;

// Write records
for record in reader {
    let record = record?;
    writer.write_record(&record)?;
}

// With compression
let mut writer = FastqWriter::from_path_with_compression(
    "output.fastq.gz",
    Compression::Gzip
)?;
```

### Batch Processing

```rust
use deepbiop_fq::dataset::FastqDataset;

// Create dataset with chunking
let dataset = FastqDataset::new("large.fastq", 10000)?;

// Process in chunks
for chunk in dataset.iter_chunks() {
    let records = chunk?;
    println!("Processing {} records", records.len());

    // Parallel processing with rayon
    use rayon::prelude::*;
    records.par_iter().for_each(|record| {
        // Process record in parallel
    });
}
```

## FASTA Processing

### Reading FASTA Files

```rust
use deepbiop_fa::reader::FastaReader;

// Read FASTA
let reader = FastaReader::from_path("genome.fasta")?;

for result in reader {
    let record = result?;
    println!(">{}", record.id());
    println!("{}", String::from_utf8_lossy(record.sequence()));
}

// With compression
let reader = FastaReader::from_path("genome.fasta.gz")?;
```

### Writing FASTA Files

```rust
use deepbiop_fa::writer::FastaWriter;

let mut writer = FastaWriter::from_path("output.fasta")?;

for record in reader {
    let record = record?;
    writer.write_record(&record)?;
}
```

### Format Conversion

```rust
use deepbiop_fq::reader::FastqReader;
use deepbiop_fa::writer::FastaWriter;

// FASTQ → FASTA
let reader = FastqReader::from_path("input.fastq")?;
let mut writer = FastaWriter::from_path("output.fasta")?;

for result in reader {
    let fastq_record = result?;

    // Convert to FASTA (drop quality scores)
    let fasta_record = fastq_record.to_fasta();
    writer.write_record(&fasta_record)?;
}
```

## BAM/SAM Processing

### Reading BAM Files

```rust
use deepbiop_bam::reader::BamReader;

// Open BAM file
let mut reader = BamReader::from_path("alignments.bam")?;

// Read header
let header = reader.header();
println!("Reference sequences: {}", header.reference_sequences().len());

// Iterate over records
for result in reader.records() {
    let record = result?;
    println!("Read: {}, MAPQ: {}", record.name(), record.mapping_quality());
}
```

### Counting Chimeric Reads

```rust
use deepbiop_bam::chimeric::ChimericCounter;

// Count chimeric reads
let counter = ChimericCounter::from_path("alignments.bam")?;
let count = counter.count()?;

println!("Chimeric reads: {}", count);
```

### BAM to FASTQ Conversion

```rust
use deepbiop_bam::reader::BamReader;
use deepbiop_fq::writer::FastqWriter;

// Convert BAM → FASTQ
let reader = BamReader::from_path("input.bam")?;
let mut writer = FastqWriter::from_path("output.fastq")?;

for result in reader.to_fastq_records() {
    let fastq_record = result?;
    writer.write_record(&fastq_record)?;
}
```

## VCF/GTF Processing

### VCF Variant Filtering

```rust
use deepbiop_vcf::reader::VcfReader;

// Read VCF
let reader = VcfReader::from_path("variants.vcf")?;

// Filter by quality
let high_quality: Vec<_> = reader
    .records()
    .filter_map(|r| r.ok())
    .filter(|variant| {
        variant.quality().map_or(false, |q| q >= 30.0)
    })
    .collect();

println!("High-quality variants: {}", high_quality.len());
```

### GTF Annotation Queries

```rust
use deepbiop_gtf::reader::GtfReader;

// Read GTF
let reader = GtfReader::from_path("annotations.gtf")?;

// Filter genes
let genes: Vec<_> = reader
    .records()
    .filter_map(|r| r.ok())
    .filter(|record| record.feature_type() == "gene")
    .collect();

println!("Total genes: {}", genes.len());
```

## Encoding for ML

DeepBioP provides three encoding strategies optimized for different ML models.

### One-Hot Encoding (CNNs/RNNs)

```rust
use deepbiop_fq::encoder::{OneHotEncoder, EncodingType, UnknownStrategy};
use ndarray::Array2;

// Create encoder
let encoder = OneHotEncoder::new(
    EncodingType::DNA,
    UnknownStrategy::Skip
);

// Encode single sequence
let sequence = b"ACGT";
let encoded: Array2<f32> = encoder.encode(sequence)?;
assert_eq!(encoded.shape(), &[4, 4]);  // seq_len × channels

// Batch encoding (parallelized)
let sequences = vec![b"ACGT".as_slice(), b"TTGG".as_slice()];
let batch_encoded = encoder.encode_batch(&sequences)?;
assert_eq!(batch_encoded.shape(), &[2, 4, 4]);  // batch × seq_len × channels
```

### Integer Encoding (Transformers/Embeddings)

```rust
use deepbiop_fq::encoder::{IntegerEncoder, EncodingType};

let encoder = IntegerEncoder::new(
    EncodingType::DNA,
    UnknownStrategy::Skip
);

let sequence = b"ACGT";
let encoded = encoder.encode(sequence)?;
// [0, 1, 2, 3] for A, C, G, T

// Use for embedding layers
println!("Encoded indices: {:?}", encoded);
```

### K-mer Encoding (Feature-Based Models)

```rust
use deepbiop_core::kmer::KmerEncoder;

// 3-mer encoding
let encoder = KmerEncoder::new(3, false)?;  // k=3, canonical=false

let sequence = b"ACGTACGT";
let kmers = encoder.encode(sequence)?;

// K-mer frequencies for Random Forest, XGBoost
println!("K-mer vector length: {}", kmers.len());
```

## Data Augmentation

### Reverse Complement

```rust
use deepbiop_fq::augmentation::ReverseComplement;

let augment = ReverseComplement::new();

let sequence = b"ACGT";
let rev_comp = augment.apply(sequence)?;
assert_eq!(rev_comp, b"ACGT");  // Reverse complement
```

### Random Mutations

```rust
use deepbiop_fq::augmentation::Mutator;

// 1% mutation rate
let mut mutator = Mutator::new(0.01, Some(42));  // seed=42

let sequence = b"ACGTACGT".repeat(10);  // 80bp
let mutated = mutator.apply(&sequence)?;

// ~0.8 bases will be mutated
```

### Subsequence Sampling

```rust
use deepbiop_fq::augmentation::Sampler;

// Random 100bp window
let mut sampler = Sampler::new(100, SamplingMode::Random, Some(42));

let sequence = b"ACGT".repeat(50);  // 200bp
let window = sampler.apply(&sequence)?;
assert_eq!(window.len(), 100);

// Different modes
let sampler_start = Sampler::new(100, SamplingMode::Start, None);
let sampler_center = Sampler::new(100, SamplingMode::Center, None);
let sampler_end = Sampler::new(100, SamplingMode::End, None);
```

## Advanced Usage

### Parallel Processing with Rayon

```rust
use rayon::prelude::*;
use deepbiop_fq::reader::FastqReader;
use deepbiop_fq::encoder::OneHotEncoder;

// Read records
let reader = FastqReader::from_path("data.fastq")?;
let records: Vec<_> = reader.collect::<Result<Vec<_>, _>>()?;

// Parallel encoding
let encoder = OneHotEncoder::new(EncodingType::DNA, UnknownStrategy::Skip);

let encoded: Vec<_> = records
    .par_iter()
    .map(|record| encoder.encode(record.sequence()))
    .collect::<Result<Vec<_>, _>>()?;

println!("Encoded {} sequences in parallel", encoded.len());
```

### Custom Record Processing

```rust
use deepbiop_fq::record::FastqRecord;

// Custom filter
fn is_high_quality(record: &FastqRecord, min_quality: u8) -> bool {
    record.quality()
        .iter()
        .all(|&q| q - 33 >= min_quality)  // Phred+33
}

// Apply filter
let reader = FastqReader::from_path("data.fastq")?;
let high_quality: Vec<_> = reader
    .filter_map(|r| r.ok())
    .filter(|record| is_high_quality(record, 30))
    .collect();

println!("High-quality reads: {}", high_quality.len());
```

### Streaming Large Files

```rust
use deepbiop_fq::reader::FastqReader;
use std::io::BufReader;
use std::fs::File;

// Streaming with custom buffer size
let file = File::open("large.fastq")?;
let buf_reader = BufReader::with_capacity(1024 * 1024, file);  // 1MB buffer
let reader = FastqReader::new(buf_reader);

// Process with constant memory
for result in reader {
    let record = result?;
    // Process record immediately (not stored in memory)
}
```

### Error Handling

```rust
use deepbiop_fq::reader::FastqReader;
use deepbiop_fq::error::FastqError;

match FastqReader::from_path("data.fastq") {
    Ok(reader) => {
        for result in reader {
            match result {
                Ok(record) => {
                    // Process record
                }
                Err(FastqError::InvalidFormat(msg)) => {
                    eprintln!("Invalid format: {}", msg);
                }
                Err(FastqError::Io(e)) => {
                    eprintln!("I/O error: {}", e);
                }
                Err(e) => {
                    eprintln!("Error: {}", e);
                }
            }
        }
    }
    Err(e) => {
        eprintln!("Failed to open file: {}", e);
    }
}
```

### Working with ndarray

```rust
use ndarray::{Array2, Array3};
use deepbiop_fq::encoder::OneHotEncoder;

// Single sequence encoding
let encoder = OneHotEncoder::new(EncodingType::DNA, UnknownStrategy::Skip);
let encoded: Array2<f32> = encoder.encode(b"ACGT")?;

// Access data
for (i, row) in encoded.axis_iter(ndarray::Axis(0)).enumerate() {
    println!("Position {}: {:?}", i, row);
}

// Convert to raw vec
let flat: Vec<f32> = encoded.into_raw_vec();

// Reshape for batches
let batch: Array3<f32> = Array3::from_shape_vec((2, 4, 4), flat)?;
```

### Building a Data Pipeline

```rust
use deepbiop_fq::reader::FastqReader;
use deepbiop_fq::encoder::OneHotEncoder;
use deepbiop_fq::augmentation::{ReverseComplement, Mutator};

struct DataPipeline {
    encoder: OneHotEncoder,
    rev_comp: ReverseComplement,
    mutator: Mutator,
}

impl DataPipeline {
    fn new() -> Self {
        Self {
            encoder: OneHotEncoder::new(EncodingType::DNA, UnknownStrategy::Skip),
            rev_comp: ReverseComplement::new(),
            mutator: Mutator::new(0.01, Some(42)),
        }
    }

    fn process(&mut self, sequence: &[u8]) -> Result<Array2<f32>, Box<dyn std::error::Error>> {
        // Apply augmentations
        let augmented = self.mutator.apply(sequence)?;
        let augmented = self.rev_comp.apply(&augmented)?;

        // Encode
        let encoded = self.encoder.encode(&augmented)?;

        Ok(encoded)
    }
}

// Usage
let mut pipeline = DataPipeline::new();
let reader = FastqReader::from_path("data.fastq")?;

for result in reader {
    let record = result?;
    let processed = pipeline.process(record.sequence())?;
    // Use processed tensor for ML
}
```

## Performance Tips

1. **Use feature flags**: Only enable features you need to reduce compilation time

2. **Parallel processing**: Leverage `rayon` for CPU-bound operations

3. **Streaming**: Use iterators instead of collecting into Vec for large files

4. **Buffer sizes**: Adjust BufReader capacity based on your file sizes

5. **Batch operations**: Use `encode_batch()` instead of looping with `encode()`

6. **Release builds**: Always use `--release` for production performance

```bash
cargo build --release
# 10-100x faster than debug builds
```

## Integration Examples

### With Burn (Deep Learning)

```rust
use burn::tensor::{Tensor, backend::Backend};
use deepbiop_fq::encoder::OneHotEncoder;

fn prepare_training_data<B: Backend>(
    sequences: &[&[u8]]
) -> Result<Tensor<B, 3>, Box<dyn std::error::Error>> {
    let encoder = OneHotEncoder::new(EncodingType::DNA, UnknownStrategy::Skip);

    // Encode sequences
    let encoded = encoder.encode_batch(sequences)?;

    // Convert to Burn tensor
    let tensor = Tensor::<B, 3>::from_data(encoded.into_raw_vec().as_slice());

    Ok(tensor)
}
```

### With Polars (Data Analysis)

```rust
use polars::prelude::*;
use deepbiop_fq::reader::FastqReader;

fn create_dataframe(path: &str) -> Result<DataFrame, Box<dyn std::error::Error>> {
    let reader = FastqReader::from_path(path)?;

    let mut ids = Vec::new();
    let mut sequences = Vec::new();
    let mut lengths = Vec::new();

    for result in reader {
        let record = result?;
        ids.push(record.id().to_string());
        sequences.push(String::from_utf8_lossy(record.sequence()).to_string());
        lengths.push(record.sequence().len() as i64);
    }

    let df = df! {
        "id" => ids,
        "sequence" => sequences,
        "length" => lengths,
    }?;

    Ok(df)
}
```

## Testing

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use deepbiop_fq::encoder::OneHotEncoder;

    #[test]
    fn test_encoding() {
        let encoder = OneHotEncoder::new(
            EncodingType::DNA,
            UnknownStrategy::Skip
        );

        let encoded = encoder.encode(b"ACGT").unwrap();

        assert_eq!(encoded.shape(), &[4, 4]);
        assert_eq!(encoded[[0, 0]], 1.0);  // A
        assert_eq!(encoded[[1, 1]], 1.0);  // C
        assert_eq!(encoded[[2, 2]], 1.0);  // G
        assert_eq!(encoded[[3, 3]], 1.0);  // T
    }
}
```

## Next Steps

- See [API Documentation](https://docs.rs/deepbiop) for complete API reference
- Check [Python API Guide](python-api.md) for using DeepBioP from Python
- Explore [CLI Usage](../cli/cli.md) for command-line tools
- Review [Examples](https://github.com/cauliyang/DeepBioP/tree/main/examples) in the repository

## Troubleshooting

### Compilation errors

Make sure you're using the correct MSRV:

```bash
rustc --version  # Should be >= 1.90.0
```

### Link errors

Some features require system libraries:

```bash
# On Ubuntu/Debian
sudo apt-get install zlib1g-dev

# On macOS
brew install zlib
```

### Performance issues

Use release builds and check feature flags:

```bash
cargo build --release --features fastq,bam
```
