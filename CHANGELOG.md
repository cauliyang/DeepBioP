# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2025-11-03

### Added

#### Sequence Encoding
- **One-Hot Encoder**: DNA/RNA sequence encoding for CNNs/RNNs with skip/mask strategies for ambiguous bases
- **K-mer Encoder**: Frequency-based encoding for feature-based models (Random Forest, SVM, XGBoost)
- **Integer Encoder**: Compact tokenization for Transformers and embedding layers
- **Batch Processing**: Parallel encoding with rayon for high-throughput processing (1M+ sequences/sec)
- NumPy interop with zero-copy export for PyTorch, TensorFlow, and JAX

#### BAM/SAM Alignment Processing
- **BamReader**: Multithreaded BAM decompression with configurable worker count
- **AlignmentFeatures**: ML feature extraction from alignments (MAPQ, identity, indel rates, CIGAR stats)
- **Filtering**: By mapping quality, proper pairs, and alignment flags
- **Chimeric Detection**: Count supplementary/chimeric alignments for structural variant analysis
- Python bindings with comprehensive error handling

#### Data Augmentation
- **ReverseComplement**: Orientation-invariant training with batch processing
- **Mutator**: Configurable mutation rates for simulating SNPs and sequencing errors
- **Sampler**: Extract subsequences with strategies (start, center, end, random)
- **QualitySimulator**: Generate realistic Phred scores with 5 models:
  - Uniform: Flat distribution
  - Normal: Gaussian distribution
  - HighQuality: Modern Illumina (Q37 mean)
  - MediumQuality: Older platforms (Q28 mean)
  - Degrading: Linear quality decline across read
- Reproducible augmentation with seed control

#### VCF Variant Processing
- **VcfReader**: Parse VCF files with filtering capabilities
- **Variant Classification**: SNP/indel detection, multiallelic handling
- **Quality Filtering**: By MAPQ threshold and FILTER field
- Python API with pandas/Arrow export support

#### GTF/GFF Annotation Processing
- **GtfReader**: Parse GTF annotation files
- **Feature Indexing**: Build gene-based indices for fast lookups
- **Type Filtering**: Query by feature type (gene, exon, CDS, transcript)
- **Attribute Extraction**: gene_id, transcript_id, gene_name helpers
- Python bindings with dictionary-based attribute access

#### ML-Friendly Export
- **Parquet Export**: Columnar storage for pandas/polars/DuckDB analytics
- **NumPy Export**: Direct .npy file creation (one-hot and integer encodings)
- **Arrow Integration**: Zero-copy sharing with Python data ecosystem
- Automatic GC content calculation for sequences

#### Documentation & Examples
- **Jupyter Notebooks**:
  - `01_fastq_encoding.ipynb`: Encoding schemes with PyTorch/HuggingFace integration
  - `02_augmentation.ipynb`: Complete augmentation pipeline with visualizations
  - `03_bam_features.ipynb`: Alignment analysis and ML feature extraction
- **Python Tests**:
  - `test_fq_encoding.py`: 260+ lines of encoder tests
  - `test_augment.py`: 390+ lines of augmentation tests
  - `test_vcf.py`: 350+ lines of VCF parsing tests
  - `test_gtf.py`: 430+ lines of GTF query tests
- Comprehensive README with quickstart examples for all features

### Changed
- Updated README.md with comprehensive feature overview organized by category
- Enhanced Python API documentation with type hints and docstrings
- Improved error messages with file/line/field context

### Fixed
- Fixed noodles API compatibility for BAM Record trait methods returning Results
- Fixed mapping quality extraction from BAM records (Option<Result<MappingQuality>>)
- Fixed data tag iteration in BAM records to handle Result types
- Fixed rand API usage for Rust 0.9 (gen_range ’ random_range, seed-based RNG)
- Resolved borrow checker issues in quality score generation loops

### Performance
- Batch encoding operations parallelized with rayon
- Streaming I/O for memory-efficient processing of 100GB+ files
- Zero-copy NumPy array export where possible
- Multithreaded BAM decompression with bgzf

## [0.1.16] - 2024-11-02

### Added
- Initial release with FASTQ/FASTA processing
- Basic sequence I/O and filtering
- Python bindings via PyO3
- CLI tool foundation

[Unreleased]: https://github.com/cauliyang/DeepBioP/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/cauliyang/DeepBioP/compare/v0.1.16...v0.2.0
[0.1.16]: https://github.com/cauliyang/DeepBioP/releases/tag/v0.1.16
