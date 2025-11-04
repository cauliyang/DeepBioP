# Feature Specification: PyTorch-Style Python API for Deep Learning

**Feature Branch**: `002-pytorch-api`
**Created**: 2025-11-03
**Status**: Draft
**Input**: User description: "We want the project provide convenient and user friendly Python API. The design of Python API mimic PyTorch and its extension, as well as transformer. Also implement necessary features that provide complete features to help researchers use biological data in deep learning"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Intuitive Dataset Loading and Preprocessing (Priority: P1)

A computational biologist wants to load FASTQ sequencing data and prepare it for training a neural network model, similar to how they would use PyTorch's `Dataset` and `DataLoader` classes for image data.

**Why this priority**: Data loading and preprocessing is the foundation of any deep learning workflow. Without this, researchers cannot even begin their work. This must be the first priority as it's the entry point to all other functionality.

**Independent Test**: Can be fully tested by loading a FASTQ file, applying transformations (encoding, augmentation), and iterating through batches. Delivers immediate value by replacing manual data preprocessing code.

**Acceptance Scenarios**:

1. **Given** a researcher has FASTQ files, **When** they create a dataset object with file paths, **Then** the system loads and validates the data without requiring manual file parsing
2. **Given** a dataset is created, **When** the researcher applies encoding transformations (one-hot, k-mer, integer), **Then** the sequences are automatically converted to numerical arrays ready for model input
3. **Given** encoded data, **When** the researcher creates a dataloader with batch size and shuffle options, **Then** the system yields batches of data with consistent shapes suitable for neural network training
4. **Given** a dataloader, **When** the researcher iterates through it, **Then** each batch contains properly formatted tensors (sequences and optional labels/metadata)

---

### User Story 2 - Flexible Data Augmentation Pipeline (Priority: P2)

A machine learning researcher wants to apply biological sequence augmentations (reverse complement, mutations, sampling) to increase training data diversity, using a composable pipeline similar to torchvision transforms.

**Why this priority**: Data augmentation is critical for training robust models, especially with limited biological data. This builds on P1's data loading capability and directly improves model performance.

**Independent Test**: Can be tested by applying single or chained transformations to sequences and verifying output correctness. Delivers value by automatically generating varied training examples.

**Acceptance Scenarios**:

1. **Given** a dataset of DNA sequences, **When** the researcher composes multiple transformations (reverse complement + mutation + sampling), **Then** the pipeline applies them in order to each sequence
2. **Given** augmentation parameters (mutation rate, sample length), **When** transformations are applied, **Then** the output maintains biological validity (valid nucleotides, correct sequence length)
3. **Given** a transformation pipeline, **When** integrated with a dataloader, **Then** augmentations are applied on-the-fly during batch generation without pre-computing all variants
4. **Given** reproducibility requirements, **When** a researcher sets a random seed, **Then** augmentation produces identical results across runs

---

### User Story 3 - Model-Ready Tensor Operations (Priority: P3)

A deep learning engineer wants to work with biological sequence data using familiar tensor operations (indexing, slicing, concatenation) that integrate seamlessly with PyTorch models.

**Why this priority**: Enables researchers to use their existing PyTorch knowledge without learning new APIs. Makes the library a drop-in replacement for biological data.

**Independent Test**: Can be tested by performing common tensor operations on biological data and verifying PyTorch compatibility. Delivers value by reducing cognitive load and code complexity.

**Acceptance Scenarios**:

1. **Given** an encoded sequence batch, **When** the researcher indexes or slices the batch, **Then** the operation returns properly shaped tensors compatible with PyTorch operations
2. **Given** multiple encoded batches, **When** the researcher concatenates them, **Then** the result maintains data integrity and works with PyTorch models
3. **Given** encoded biological data, **When** passed directly to a PyTorch model, **Then** the forward pass executes without type or shape errors
4. **Given** model outputs, **When** the researcher applies loss functions, **Then** standard PyTorch loss functions work without custom wrappers

---

### User Story 4 - Convenient Export and Persistence (Priority: P4)

A researcher wants to save processed datasets in efficient formats (Parquet, HDF5, PyTorch tensor files) for reuse across experiments without re-processing raw data every time.

**Why this priority**: Supports efficient workflow by eliminating redundant preprocessing. Lower priority as it's an optimization rather than core functionality.

**Independent Test**: Can be tested by saving processed data and reloading it in a new session. Delivers value by saving computation time in subsequent experiments.

**Acceptance Scenarios**:

1. **Given** a processed dataset, **When** the researcher saves it using the export function, **Then** the data is written to disk in the specified format with metadata
2. **Given** an exported dataset file, **When** loaded in a new Python session, **Then** the data loads quickly without re-processing and maintains original transformations
3. **Given** large datasets, **When** exporting to Parquet/HDF5, **Then** the storage size is optimized through compression without data loss
4. **Given** collaborative requirements, **When** datasets are shared, **Then** other researchers can load them without access to original raw files

---

### User Story 5 - Informative Dataset Inspection and Validation (Priority: P5)

A researcher wants to inspect dataset properties (sequence lengths, class distribution, encoding statistics) and validate data quality before starting expensive training runs.

**Why this priority**: Helps catch data issues early but is supplementary to core functionality. Can be implemented after basic workflows are operational.

**Independent Test**: Can be tested by calling inspection methods and verifying returned statistics match expected values. Delivers value by preventing wasted computation on bad data.

**Acceptance Scenarios**:

1. **Given** a loaded dataset, **When** the researcher calls summary methods, **Then** the system reports total samples, sequence length distribution, and memory footprint
2. **Given** a dataset with labels, **When** requesting class distribution, **Then** the system shows balanced/imbalanced classes with counts and percentages
3. **Given** encoded data, **When** the researcher validates it, **Then** the system checks for NaN values, inf values, and out-of-range values
4. **Given** quality issues, **When** validation runs, **Then** the system provides actionable warnings (e.g., "50% of sequences contain ambiguous nucleotides")

---

### Edge Cases

- **Empty or corrupted files**: What happens when a FASTQ file is empty, truncated, or contains invalid format?
- **Memory constraints**: How does the system handle datasets too large to fit in memory during batch generation?
- **Inconsistent sequence lengths**: How are variable-length sequences handled in batching (padding, truncation, dynamic batching)?
- **Invalid augmentation parameters**: What happens when mutation rate > 1.0 or sample length exceeds sequence length?
- **Missing labels or metadata**: How does the system behave when optional metadata files are absent?
- **Mixed data types**: How are datasets with multiple sequence types (DNA, RNA, protein) handled?
- **Concurrent access**: What happens when multiple processes try to read/write the same cached dataset?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST provide a Dataset class that loads biological sequence data from common file formats (FASTQ, FASTA) without manual parsing
- **FR-002**: System MUST support lazy loading where sequences are read from disk only when accessed, not all at once
- **FR-003**: System MUST provide encoding methods (one-hot, integer, k-mer) accessible through dataset transformations
- **FR-004**: System MUST provide a DataLoader class that generates batches with configurable size, shuffling, and parallel loading
- **FR-005**: System MUST implement augmentation transformations (reverse complement, mutation, sampling) that can be composed in pipelines
- **FR-006**: System MUST handle variable-length sequences through automatic padding or truncation when batching
- **FR-007**: System MUST support random seeds for reproducible shuffling and augmentation
- **FR-008**: System MUST return data in formats compatible with PyTorch tensors (NumPy arrays that convert seamlessly)
- **FR-009**: System MUST provide export functions for processed datasets in Parquet and HDF5 formats
- **FR-010**: System MUST support optional quality score preservation and filtering for FASTQ data
- **FR-011**: System MUST implement caching mechanisms for commonly accessed datasets to avoid redundant disk I/O
- **FR-012**: System MUST validate sequence data (checking for invalid characters based on sequence type)
- **FR-013**: System MUST provide dataset inspection methods (length distribution, summary statistics, sample preview)
- **FR-014**: System MUST handle metadata and labels through optional companion files or embedded formats
- **FR-015**: System MUST support stratified sampling for imbalanced datasets
- **FR-016**: System MUST allow custom transformation functions defined by users
- **FR-017**: System MUST implement collate functions for batching that handle sequences, labels, and metadata together
- **FR-018**: System MUST provide progress indicators for long-running operations (loading, encoding, export)
- **FR-019**: System MUST support both single-sequence and paired-end sequence data (for paired FASTQ files)
- **FR-020**: System MUST implement memory-efficient batch generation using iterators rather than loading all data upfront

### Key Entities

- **Dataset**: Represents a collection of biological sequences with optional labels and metadata. Provides indexing, length, and transformation capabilities. Can be created from file paths or in-memory data.
- **DataLoader**: Manages batch generation from a Dataset with shuffling, parallel loading (using multiple workers), and custom collate functions. Similar to torch.utils.data.DataLoader.
- **Transform**: Represents a data transformation operation (encoding, augmentation). Can be a single operation or a composition pipeline. Applied to individual samples or batches.
- **Encoder**: Converts biological sequences to numerical representations (one-hot, integer, k-mer). Each encoder type has specific parameters (k-mer size, canonical mode, unknown base handling).
- **Augmentation**: Biological sequence transformations that preserve validity (reverse complement, mutation, subsequence sampling). Each has configurable parameters (mutation rate, sample length, random seed).
- **Sample**: A single data point containing a sequence (string or bytes), optional quality scores, optional label, and optional metadata dictionary.
- **Batch**: A collection of samples formatted for model input, with padded/truncated sequences of uniform length, stacked labels, and combined metadata.
- **Cache**: Persistent storage of processed datasets to avoid recomputation. Tracks dataset version, transformations applied, and source file modification times.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Researchers can load a standard FASTQ dataset and create batches ready for training in under 5 lines of code
- **SC-002**: Data loading and batching overhead is under 10% of total training time for typical workflows
- **SC-003**: 90% of researchers familiar with PyTorch DataLoader can use the biological data API without reading documentation
- **SC-004**: The API supports training loops with 1000+ epochs without memory leaks or performance degradation
- **SC-005**: Batch generation with augmentation processes at least 10,000 sequences per second on standard hardware
- **SC-006**: Dataset inspection and validation completes in under 1 second per 10,000 sequences
- **SC-007**: Cached datasets load 10x faster than processing from raw files
- **SC-008**: The API reduces typical data preprocessing code from 100+ lines to under 20 lines
- **SC-009**: Users report 80%+ satisfaction with API intuitiveness in user testing
- **SC-010**: Integration with existing PyTorch models requires zero code changes to the model architecture

### Non-Functional Outcomes

- API naming conventions match PyTorch standards (snake_case for functions, CamelCase for classes)
- Error messages provide actionable guidance (e.g., "Invalid nucleotide 'X' at position 42, expected A/C/G/T")
- Documentation includes runnable examples for all common workflows
- Type hints are provided for all public APIs to enable IDE autocomplete
- Memory usage scales linearly with batch size, not total dataset size

## Assumptions

1. **Target users**: Researchers familiar with PyTorch who want to work with biological sequence data
2. **Primary use case**: Supervised learning on pre-labeled sequence datasets
3. **Data scale**: Datasets ranging from thousands to millions of sequences, individual files < 50GB
4. **Hardware**: Standard research workstations with 16-64GB RAM, 4-16 CPU cores
5. **Integration**: Primarily used with PyTorch deep learning framework, secondary TensorFlow compatibility
6. **File formats**: Standard bioinformatics formats (FASTQ, FASTA) are sufficient; no need for exotic formats
7. **Performance baseline**: Similar or better performance than existing bioinformatics Python libraries (Biopython, pysam)
8. **Quality scores**: Phred+33 encoding is the standard for FASTQ quality scores
9. **Augmentation philosophy**: Augmentations preserve biological plausibility (valid sequences, realistic quality degradation)
10. **Caching strategy**: Disk-based caching is preferred over memory caching due to dataset sizes

## Constraints

1. **Backward compatibility**: Must maintain compatibility with existing DeepBioP core functionality
2. **Performance**: Cannot be significantly slower than direct file reading for simple use cases
3. **Memory efficiency**: Must support datasets larger than available RAM through streaming/chunking
4. **Cross-platform**: Must work on Linux, macOS, and Windows
5. **Python versions**: Must support Python 3.9-3.12
6. **Dependencies**: Should minimize new dependencies; prefer extending existing libraries (NumPy, PyTorch)

## Dependencies

- Existing DeepBioP encoding and augmentation functionality (one-hot, k-mer, integer encoders)
- Existing file I/O capabilities for FASTQ, FASTA, Parquet
- NumPy for array operations
- PyTorch (optional but primary integration target)
- Standard Python libraries (itertools, pathlib, multiprocessing)

## Out of Scope

- Real-time data streaming from sequencing instruments
- Distributed data loading across multiple machines
- Automatic hyperparameter tuning for augmentation parameters
- Custom file format parsers beyond standard bioinformatics formats
- GPU-accelerated data preprocessing (stays on CPU)
- Built-in model architectures for sequence analysis
- Training loop utilities (epochs, checkpointing, logging) - users will use PyTorch Lightning or similar
- Visualization of sequences or training progress
- Integration with workflow engines (Nextflow, Snakemake)
- Cloud storage integration (S3, GCS) - users handle this externally
