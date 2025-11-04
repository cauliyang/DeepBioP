# Data Model: PyTorch-Style Python API

**Date**: 2025-11-03
**Feature**: PyTorch-Style Python API for Deep Learning
**Phase**: 1 - Design

This document defines the core entities, their attributes, relationships, and state transitions for the PyTorch-compatible data loading API.

---

## Entity Diagram

```
┌─────────────────┐
│                 │
│     Dataset     │◄────────────┐
│                 │              │
└────────┬────────┘              │
         │                       │
         │ 1                     │
         │                       │
         │ *                     │ delegates to
         │                       │
         ▼                       │
┌─────────────────┐              │
│                 │              │
│   DataLoader    │──────────────┤
│                 │              │
└────────┬────────┘              │
         │                       │
         │ uses                  │
         │                       │
         ▼                       │
┌─────────────────┐              │
│                 │              │
│  Collate Fn     │──────────────┤
│                 │              │
└─────────────────┘              │
         │                       │
         │                       │
         ▼                       │
┌─────────────────┐              │
│                 │              │
│    Transform    │──────────────┤
│                 │              │
└────────┬────────┘              │
         │                       │
         ├─ Encoder ─────────────┤
         │                       │
         └─ Augmentation ────────┤
                                 │
┌─────────────────┐              │
│                 │              │
│     Cache       │──────────────┘
│                 │
└─────────────────┘

Legend:
─── : has-a / contains
──► : delegates to / uses
```

---

## Entity Definitions

### 1. Dataset

**Purpose**: Wraps biological sequence files (FASTQ/FASTA) for random access and iteration.

**Attributes**:

| Name | Type | Validation | Description |
|------|------|------------|-------------|
| `file_paths` | `List[str]` | Must exist, non-empty, valid FASTQ/FASTA format | Paths to input sequence files |
| `sequence_type` | `enum: DNA \| RNA \| Protein` | Must match file content | Type of biological sequences |
| `records_count` | `int` | > 0 | Total number of sequences (computed on init) |
| `transform` | `Optional[Transform]` | Valid Transform instance or None | Encoding/augmentation pipeline to apply |
| `cache_dir` | `Optional[str]` | Must be writable directory or None | Directory for cached processed data |
| `lazy` | `bool` | - | If True, sequences loaded on-demand; if False, preload in memory |

**Methods**:

```python
def __init__(
    file_paths: Union[str, List[str]],
    sequence_type: str = "dna",
    transform: Optional[Transform] = None,
    cache_dir: Optional[str] = None,
    lazy: bool = True
) -> None

def __len__() -> int
    """Returns total number of sequences"""

def __getitem__(idx: int) -> Sample
    """Returns sample at index idx (applies transform if set)"""

def __iter__() -> Iterator[Sample]
    """Returns iterator over all samples"""

def summary() -> Dict[str, Any]
    """Returns dataset statistics (counts, length distribution, etc.)"""
```

**Validation Rules**:
- FR-001: Must wrap existing DeepBioP file readers (no custom parsing)
- FR-002: Must support lazy loading (default)
- FR-012: Must validate sequence data using existing DeepBioP validation

**State Transitions**:
```
[Created] ──> [Validated] ──> [Ready]
                   │              │
                   │              ├──> [Cached] (if cache hit)
                   │              └──> [Loading] ──> [Ready]
                   ▼
              [Invalid] (file not found, format error)
```

**Relationships**:
- Contains 0..1 Transform (optional encoding/augmentation)
- Delegates file reading to existing FASTQ/FASTA readers
- May use Cache for preprocessed data

---

### 2. DataLoader

**Purpose**: Generates batches of samples with shuffling, parallel loading, and collation.

**Attributes**:

| Name | Type | Validation | Description |
|------|------|------------|-------------|
| `dataset` | `Dataset` | Must be valid Dataset instance | Source dataset |
| `batch_size` | `int` | > 0 | Number of samples per batch |
| `shuffle` | `bool` | - | Randomize sample order each epoch |
| `num_workers` | `int` | >= 0 | Number of parallel loading processes (0 = main process) |
| `collate_fn` | `Optional[Callable]` | Valid callable or None | Function to merge samples into batch |
| `drop_last` | `bool` | - | Drop incomplete final batch |
| `seed` | `Optional[int]` | >= 0 or None | Random seed for reproducibility |

**Methods**:

```python
def __init__(
    dataset: Dataset,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
    collate_fn: Optional[Callable] = None,
    drop_last: bool = False,
    seed: Optional[int] = None
) -> None

def __iter__() -> Iterator[Batch]
    """Returns iterator over batches"""

def __len__() -> int
    """Returns number of batches (depends on batch_size and drop_last)"""
```

**Validation Rules**:
- FR-004: Must provide batching, shuffling, and parallel loading
- FR-007: Must support random seeds for reproducibility
- FR-020: Must use memory-efficient iterators (no full-data buffering)

**Relationships**:
- Contains exactly 1 Dataset
- Uses CollateFunction to merge samples
- Delegates to existing batching utilities (FR-017)

---

### 3. Transform

**Purpose**: Abstract base class for sequence transformations (encoding, augmentation).

**Attributes**:

| Name | Type | Validation | Description |
|------|------|------------|-------------|
| `transforms` | `List[Transform]` | All must be valid Transform instances | Ordered list of transformations |

**Methods**:

```python
def __call__(sample: Sample) -> Sample
    """Apply transformation to a single sample"""

def __repr__() -> str
    """Human-readable representation"""
```

**Subclasses**:

#### 3a. Encoder (Transform subclass)

**Purpose**: Wraps existing DeepBioP encoders for sequence-to-numerical conversion.

**Variants**:

```python
class OneHotEncoder(Transform):
    """Wraps deepbiop.fq.OneHotEncoder"""
    def __init__(
        encoding_type: str = "dna",
        unknown_strategy: str = "skip"
    )

class IntegerEncoder(Transform):
    """Wraps deepbiop.fq.IntegerEncoder"""
    def __init__(encoding_type: str = "dna")

class KmerEncoder(Transform):
    """Wraps deepbiop.core.KmerEncoder"""
    def __init__(
        k: int,
        canonical: bool = False,
        encoding_type: str = "dna"
    )
```

**Validation Rules**:
- FR-003: Must delegate to existing DeepBioP encoders
- FR-021: Must NOT duplicate encoding logic

**Relationships**:
- Wraps existing `deepbiop.fq.OneHotEncoder`, `IntegerEncoder`, `KmerEncoder`
- Delegates all computation to wrapped encoder

#### 3b. Augmentation (Transform subclass)

**Purpose**: Wraps existing DeepBioP augmentations for sequence modifications.

**Variants**:

```python
class ReverseComplement(Transform):
    """Wraps deepbiop.fq.ReverseComplement"""
    def __init__(is_rna: bool = False)

class Mutator(Transform):
    """Wraps deepbiop.fq.Mutator"""
    def __init__(
        mutation_rate: float,
        seed: Optional[int] = None,
        is_rna: bool = False
    )

class Sampler(Transform):
    """Wraps deepbiop.fq.Sampler"""
    def __init__(
        length: int,
        strategy: str = "random",  # "start" | "center" | "end" | "random"
        seed: Optional[int] = None
    )
```

**Validation Rules**:
- FR-005: Must delegate to existing DeepBioP augmentations
- FR-021: Must NOT duplicate augmentation logic
- Mutation rate must be 0.0-1.0
- Sampler length must be > 0 and <= sequence length

**Relationships**:
- Wraps existing `deepbiop.fq.ReverseComplement`, `Mutator`, `Sampler`
- Delegates all computation to wrapped augmentation

---

### 4. Compose

**Purpose**: Chains multiple transforms into a pipeline.

**Attributes**:

| Name | Type | Validation | Description |
|------|------|------------|-------------|
| `transforms` | `List[Transform]` | Non-empty, all valid Transforms | Ordered transformations |

**Methods**:

```python
def __init__(transforms: List[Transform])

def __call__(sample: Sample) -> Sample:
    """Apply all transforms in sequence"""
```

**Implementation**:
```python
def __call__(self, sample):
    for t in self.transforms:
        sample = t(sample)
    return sample
```

**Validation Rules**:
- FR-005: Composition uses standard Python function chaining
- SC-014: Under 20 lines of code, no custom orchestration

**Relationships**:
- Contains 1..* Transform instances
- No delegation to existing code (simple function chaining only)

---

### 5. Collate Function

**Purpose**: Merges list of samples into a batch (padding, stacking, metadata aggregation).

**Signature**:

```python
def collate_fn(batch: List[Sample]) -> Batch:
    """
    Args:
        batch: List of samples from Dataset.__getitem__()

    Returns:
        Batch dict with keys:
            - "sequences": np.ndarray of shape [batch_size, max_len, features]
            - "labels": Optional[np.ndarray] of shape [batch_size]
            - "metadata": Optional[List[Dict]] of sample metadata
    """
```

**Variants**:

```python
def default_collate_fn(batch: List[Sample]) -> Batch:
    """Default collation: pad sequences, stack arrays"""

def pad_collate_fn(
    batch: List[Sample],
    padding_value: float = 0.0,
    pad_to: Optional[int] = None
) -> Batch:
    """Custom padding strategy"""

def truncate_collate_fn(
    batch: List[Sample],
    max_length: int
) -> Batch:
    """Truncate sequences to max_length"""
```

**Validation Rules**:
- FR-017: Must delegate to existing batching/padding utilities
- SC-012: Under 50 lines, delegates batching logic
- FR-022: Users can provide custom collate functions

**Relationships**:
- Uses existing DeepBioP batching utilities (if available)
- Otherwise uses NumPy standard functions (np.pad, np.stack)

---

### 6. Sample

**Purpose**: Represents a single data point from the dataset.

**Attributes**:

| Name | Type | Validation | Description |
|------|------|------------|-------------|
| `sequence` | `Union[bytes, np.ndarray]` | Non-empty | Raw sequence (bytes) or encoded (ndarray) |
| `quality` | `Optional[bytes]` | Same length as sequence or None | Phred+33 quality scores (FASTQ only) |
| `label` | `Optional[Union[int, float, str]]` | - | Optional classification label |
| `metadata` | `Optional[Dict[str, Any]]` | - | Optional metadata (ID, source file, etc.) |

**Methods**:

```python
def __init__(
    sequence: Union[bytes, np.ndarray],
    quality: Optional[bytes] = None,
    label: Optional[Any] = None,
    metadata: Optional[Dict] = None
)
```

**Validation Rules**:
- Sequence must be non-empty
- Quality (if provided) must match sequence length
- FR-010: Must support quality score preservation

**State Transitions**:
```
[Raw] ──> [Transformed] (if Dataset has transform)
   │           │
   └───────────┴──> [Batched] (collate_fn applied)
```

---

### 7. Batch

**Purpose**: Collection of samples formatted for model input.

**Attributes**:

| Name | Type | Validation | Description |
|------|------|------------|-------------|
| `sequences` | `np.ndarray` | Shape [batch_size, max_len, features] | Padded/truncated sequence batch |
| `labels` | `Optional[np.ndarray]` | Shape [batch_size] or None | Stacked labels |
| `quality` | `Optional[np.ndarray]` | Shape [batch_size, max_len] or None | Stacked quality scores |
| `metadata` | `Optional[List[Dict]]` | Length = batch_size or None | Per-sample metadata |
| `lengths` | `Optional[np.ndarray]` | Shape [batch_size] or None | Original sequence lengths (before padding) |

**Validation Rules**:
- All arrays must have matching batch_size in first dimension
- FR-006: Must handle variable-length sequences through padding/truncation
- FR-008: Must return PyTorch-compatible NumPy arrays

**Relationships**:
- Created by CollateFunction from List[Sample]
- Delegates padding/stacking to existing utilities or NumPy

---

### 8. Cache

**Purpose**: Persistent storage of processed datasets for faster reloading.

**Attributes**:

| Name | Type | Validation | Description |
|------|------|------------|-------------|
| `cache_dir` | `str` | Must be writable directory | Directory for cache files |
| `dataset_hash` | `str` | - | Hash of source files + transformations |
| `file_path` | `str` | - | Path to cached Parquet/HDF5 file |
| `metadata_path` | `str` | - | Path to metadata JSON (hash, timestamps, transforms) |
| `created_at` | `datetime` | - | Cache creation timestamp |
| `source_mtime` | `Dict[str, float]` | - | Source file modification times (for invalidation) |

**Methods**:

```python
def save(dataset: Dataset, cache_dir: str) -> Cache:
    """Export dataset to Parquet, write metadata"""

def load(cache_dir: str, dataset_hash: str) -> Optional[Dataset]:
    """Load from cache if valid, else None"""

def is_valid() -> bool:
    """Check if cache is still valid (source files unchanged)"""

def invalidate() -> None:
    """Delete cache files"""
```

**Validation Rules**:
- FR-011: Must use existing Parquet/HDF5 export as storage format
- SC-013: Zero custom serialization code, only metadata tracking
- FR-009: Must delegate to existing export functionality

**Cache Invalidation Logic**:
```python
def is_valid(cache: Cache, source_files: List[str]) -> bool:
    # Cache invalid if:
    # 1. Source file modified (mtime changed)
    # 2. Transform parameters changed (hash mismatch)
    # 3. DeepBioP version changed
    for file in source_files:
        if os.path.getmtime(file) != cache.source_mtime[file]:
            return False
    return True
```

**Relationships**:
- Uses existing `deepbiop.fq.write_parquet()` or similar for export
- Adds only metadata file (JSON) for hash and timestamp tracking

---

## Validation Summary by Requirement

| Requirement | Entities | Validation Rule |
|-------------|----------|-----------------|
| FR-001 | Dataset | Wraps existing file readers, no custom parsing |
| FR-002 | Dataset | Supports lazy loading (default) |
| FR-003 | Encoder | Delegates to existing DeepBioP encoders |
| FR-004 | DataLoader | Batching, shuffling, parallel loading |
| FR-005 | Augmentation | Delegates to existing DeepBioP augmentations |
| FR-006 | Batch | Handles variable-length via padding/truncation |
| FR-007 | Dataset, DataLoader | Supports random seeds |
| FR-008 | Batch | Returns NumPy arrays (PyTorch-compatible) |
| FR-009 | Cache | Delegates to existing Parquet/HDF5 export |
| FR-010 | Sample | Preserves quality scores |
| FR-011 | Cache | Uses Parquet/HDF5 + metadata tracking |
| FR-012 | Dataset | Validates using existing DeepBioP validation |
| FR-013 | Dataset | Provides summary() using NumPy functions |
| FR-017 | CollateFunction | Delegates to existing batching utilities |
| FR-021 | All Transforms | No duplication of encoder/augmentation logic |
| FR-022 | DataLoader | Accepts custom collate_fn parameter |

---

## Memory Flow Diagram

```
File (disk)
    │
    ▼
FileReader (existing)  ──────► validates format
    │
    ▼
Dataset.__getitem__(idx)
    │
    ├──► [if lazy] read sequence from disk
    ├──► [if cached] load from Parquet
    └──► [if in-memory] retrieve from memory
    │
    ▼
Transform.__call__(sample)  ──► delegates to existing encoders/augmentations
    │                             (releases GIL via py.detach() for batches)
    ▼
Sample (bytes or ndarray)
    │
    ▼
DataLoader (collects samples)
    │
    ▼
CollateFunction(batch: List[Sample])  ──► pads/stacks using NumPy or existing utils
    │
    ▼
Batch (ndarray with batch dimension)
    │
    ▼
PyTorch Model (via torch.from_numpy, zero-copy)
```

---

## Performance Considerations

1. **Lazy Loading**: Dataset loads sequences on-demand to minimize memory footprint (FR-002)
2. **Parallel Encoding**: Batch transforms release GIL for Rayon parallelism (research.md Decision 2)
3. **Zero-Copy NumPy→PyTorch**: `torch.from_numpy()` shares memory (no copy)
4. **Cache Hits**: Cached datasets load ~10x faster (SC-007) via Parquet columnar format
5. **Streaming I/O**: No full-file buffering, uses iterators (FR-020)

---

## Next Artifacts

- **contracts/api-contracts.md**: Python API signatures with type annotations
- **quickstart.md**: Example usage demonstrating all entities working together
