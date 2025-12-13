# API Reference

Complete reference for DeepBioP's Python API.

## Table of Contents

- [Datasets](#datasets)
  - [FASTQ](#fastq-datasets)
  - [FASTA](#fasta-datasets)
  - [BAM](#bam-datasets)
- [Transforms](#transforms)
  - [Filters](#filters)
  - [Augmentations](#augmentations)
  - [Encoders](#encoders)
  - [Composition](#composition)
- [PyTorch Integration](#pytorch-integration)
- [Core Utilities](#core-utilities)

---

## Datasets

### FASTQ Datasets

#### `FastqStreamDataset`

Streaming dataset for FASTQ files (plain, gzipped, or bgzipped).

```python
from deepbiop.fq import FastqStreamDataset

dataset = FastqStreamDataset(file_path: str | Path)
```

**Parameters:**
- `file_path` (str | Path): Path to FASTQ file (.fastq, .fq, .fastq.gz, .fq.gz, .fastq.bgz)

**Returns:**
Iterator yielding dict records with keys:
- `id` (str): Sequence identifier
- `sequence` (numpy.ndarray): Sequence as uint8 array
- `quality` (numpy.ndarray): Quality scores as uint8 array
- `description` (str | None): Optional description

**Example:**
```python
dataset = FastqStreamDataset("reads.fastq.gz")
for record in dataset:
    print(f"{record['id']}: {len(record['sequence'])} bases")
```

**Pickling Support:** ✅ Yes (for multiprocessing with DataLoader)

**Memory Usage:** Constant (streaming)

---

#### `FastqDataset`

Random-access FASTQ dataset (loads index for seeking).

```python
from deepbiop.fq import FastqDataset

dataset = FastqDataset(file_path: str | Path)
```

**Parameters:**
- `file_path` (str | Path): Path to FASTQ file

**Returns:**
- Supports `len()` and indexing: `dataset[0]`, `dataset[10:20]`

**Example:**
```python
dataset = FastqDataset("reads.fastq")
print(f"Total records: {len(dataset)}")
first_record = dataset[0]
```

---

### FASTA Datasets

#### `FastaStreamDataset`

Streaming dataset for FASTA files.

```python
from deepbiop.fa import FastaStreamDataset

dataset = FastaStreamDataset(file_path: str | Path)
```

**Parameters:**
- `file_path` (str | Path): Path to FASTA file (.fasta, .fa, .fasta.gz, .fa.gz)

**Returns:**
Iterator yielding dict records with keys:
- `id` (str): Sequence identifier
- `sequence` (numpy.ndarray): Sequence as uint8 array
- `description` (str | None): Optional description
- `quality` (None): Always None for FASTA

**Example:**
```python
dataset = FastaStreamDataset("genome.fasta.gz")
for record in dataset:
    print(f">{record['id']}\n{record['sequence'].tobytes().decode()}")
```

---

#### `FastaDataset`

PyTorch-compatible FASTA dataset with random access and caching.

```python
from deepbiop import FastaDataset

dataset = FastaDataset(file_path: str)
```

**Parameters:**
- `file_path` (str): Path to FASTA file (.fasta, .fa, .fasta.gz, .fa.gz)

**Features:**
- Full PyTorch Dataset protocol: `__len__`, `__getitem__`, `__iter__`
- Random access via indexing
- Caches records for efficient random access
- Compatible with PyTorch DataLoader

**Returns:**
Dict records with keys:
- `id` (bytes): Sequence identifier
- `sequence` (bytes): Sequence data
- `description` (bytes | None): Optional description

**Example:**
```python
from deepbiop import FastaDataset, default_collate
from torch.utils.data import DataLoader

# Create dataset
dataset = FastaDataset("genome.fasta.gz")

# Access records
print(f"Total: {len(dataset)}")      # Supports len()
first = dataset[0]                    # Supports indexing
for record in dataset:                # Supports iteration
    process(record)

# Use with DataLoader
loader = DataLoader(
    dataset,
    batch_size=16,
    collate_fn=default_collate  # For variable-length sequences
)

for batch in loader:
    # batch is a list of dicts
    for record in batch:
        print(record['id'], len(record['sequence']))
```

**Memory Usage:** Loads all records into memory on initialization

**Pickling Support:** ✅ Yes

---

### BAM Datasets

#### `BamStreamDataset`

Streaming dataset for BAM/SAM alignment files.

```python
from deepbiop.bam import BamStreamDataset

dataset = BamStreamDataset(
    file_path: str | Path,
    threads: int = 1,
    reference_path: str | Path | None = None
)
```

**Parameters:**
- `file_path` (str | Path): Path to BAM file
- `threads` (int, optional): Number of decompression threads (default: 1)
- `reference_path` (str | Path | None, optional): Path to reference FASTA (for CRAM files)

**Returns:**
Iterator yielding dict records with keys:
- `id` (str): Read name
- `sequence` (numpy.ndarray): Sequence as uint8 array
- `quality` (numpy.ndarray): Quality scores as uint8 array
- `flag` (int): SAM flags
- `reference_name` (str | None): Reference contig name
- `position` (int): 0-based leftmost position
- `mapping_quality` (int): Mapping quality score
- `cigar` (str): CIGAR string

**Example:**
```python
dataset = BamStreamDataset("alignments.bam", threads=4)
for record in dataset:
    if record['mapping_quality'] >= 30:
        process_alignment(record)
```

---

#### `BamDataset`

PyTorch-compatible BAM dataset with random access and multithreaded decompression.

```python
from deepbiop import BamDataset

dataset = BamDataset(
    file_path: str,
    threads: int | None = None
)
```

**Parameters:**
- `file_path` (str): Path to BAM file
- `threads` (int | None, optional): Number of threads for BGZF decompression (None = use all available)

**Features:**
- Full PyTorch Dataset protocol: `__len__`, `__getitem__`, `__iter__`
- Random access via indexing
- Multithreaded BGZF decompression for improved performance
- Caches records for efficient random access
- Compatible with PyTorch DataLoader

**Returns:**
Dict records with keys:
- `id` (bytes): Read name
- `sequence` (bytes): Sequence data
- `quality` (bytes): Quality scores
- `description` (bytes | None): Optional description

**Example:**
```python
from deepbiop import BamDataset, default_collate
from torch.utils.data import DataLoader

# Create dataset with 4 decompression threads
dataset = BamDataset("alignments.bam", threads=4)

# Access records
print(f"Total: {len(dataset)}")      # Supports len()
first = dataset[0]                    # Supports indexing
for record in dataset:                # Supports iteration
    process(record)

# Use with DataLoader
loader = DataLoader(
    dataset,
    batch_size=32,
    collate_fn=default_collate  # For variable-length sequences
)

for batch in loader:
    # batch is a list of dicts
    for record in batch:
        print(record['id'], len(record['sequence']))
```

**Performance Tips:**
- Use `threads` parameter for large BAM files (recommended: 2-8 threads)
- For small files, multithreading overhead may not improve performance

**Memory Usage:** Loads all records into memory on initialization

**Pickling Support:** ✅ Yes

---

## Transforms

### Filters

#### `QualityFilter`

Filter sequences by mean quality score.

```python
from deepbiop.fq import QualityFilter

filter = QualityFilter(
    min_mean_quality: float | None = None,
    min_base_quality: int | None = None,
    quality_offset: int = 33
)
```

**Parameters:**
- `min_mean_quality` (float, optional): Minimum mean Phred quality score
- `min_base_quality` (int, optional): Minimum quality for all bases
- `quality_offset` (int, default: 33): Phred offset (33 for Phred+33, 64 for Phred+64)

**Methods:**
- `passes(sequence: bytes, quality: bytes) -> bool`: Returns True if sequence passes filter
- `calculate_mean_quality(quality: bytes) -> float`: Calculate mean quality score

**Example:**
```python
quality_filter = QualityFilter(min_mean_quality=30.0)

for record in dataset:
    if quality_filter.passes(record['sequence'], record['quality']):
        # High-quality sequence
        process(record)
```

---

#### `LengthFilter`

Filter sequences by length.

```python
from deepbiop.fq import LengthFilter

filter = LengthFilter(
    min_length: int | None = None,
    max_length: int | None = None
)
```

**Parameters:**
- `min_length` (int, optional): Minimum sequence length
- `max_length` (int, optional): Maximum sequence length

**Methods:**
- `passes(sequence: bytes) -> bool`: Returns True if sequence length is within bounds

**Example:**
```python
length_filter = LengthFilter(min_length=50, max_length=500)

for record in dataset:
    if length_filter.passes(record['sequence']):
        # Sequence length between 50-500
        process(record)
```

---

### Augmentations

#### `Mutator`

Random point mutation augmenter for data augmentation.

```python
from deepbiop.fq import Mutator

mutator = Mutator(mutation_rate: float, seed: int | None = None)
```

**Parameters:**
- `mutation_rate` (float): Probability of mutation per base (0.0 to 1.0)
- `seed` (int, optional): Random seed for reproducibility

**Methods:**
- `augment(record: dict) -> dict`: Apply random mutations to sequence

**Example:**
```python
mutator = Mutator(mutation_rate=0.1, seed=42)

for record in dataset:
    augmented = mutator.augment(record)
    # augmented['sequence'] has ~10% of bases mutated
```

---

#### `ReverseComplement`

Reverse complement transformation for DNA/RNA sequences.

```python
from deepbiop.fq import ReverseComplement

rc = ReverseComplement()
```

**Methods:**
- `augment(record: dict) -> dict`: Reverse complement the sequence
- `for_rna() -> ReverseComplement`: Create RNA-specific reverse complement

**Example:**
```python
rc = ReverseComplement()

for record in dataset:
    rc_record = rc.augment(record)
    # rc_record['sequence'] is reverse complemented
    # ACGT → ACGT (reverse: TGCA, complement: ACGT)
```

---

### Encoders

#### `OneHotEncoder`

One-hot encode biological sequences.

```python
from deepbiop.fq import OneHotEncoder

encoder = OneHotEncoder(
    encoding_type: str = "dna",
    unknown_strategy: str = "skip"
)
```

**Parameters:**
- `encoding_type` (str): "dna" or "rna" or "protein"
- `unknown_strategy` (str): How to handle unknown bases
  - `"skip"`: Unknown bases get all zeros
  - `"error"`: Raise error on unknown bases
  - `"mask"`: Add extra dimension for unknown

**Returns:**
Encoded sequence as numpy array:
- DNA/RNA: shape `(length, 4)` for A, C, G, T/U
- Protein: shape `(length, 20)` for 20 amino acids

**Example:**
```python
encoder = OneHotEncoder(encoding_type="dna", unknown_strategy="skip")

for record in dataset:
    encoded = encoder(record)
    # encoded['sequence'].shape = (length, 4)
    # [[1,0,0,0],  # A
    #  [0,1,0,0],  # C
    #  [0,0,1,0],  # G
    #  [0,0,0,1]]  # T
```

---

#### `IntegerEncoder`

Encode sequences as integers.

```python
from deepbiop.fq import IntegerEncoder

encoder = IntegerEncoder(encoding_type: str = "dna")
```

**Parameters:**
- `encoding_type` (str): "dna" or "rna" or "protein"

**Returns:**
Encoded sequence as numpy uint8 array:
- DNA: A=0, C=1, G=2, T=3, N=4
- RNA: A=0, C=1, G=2, U=3, N=4

**Example:**
```python
encoder = IntegerEncoder(encoding_type="dna")

for record in dataset:
    encoded = encoder(record)
    # encoded['sequence'] = array([0, 1, 2, 3, ...])  # A, C, G, T, ...
```

---

#### `KmerEncoder`

Encode sequences as k-mer indices.

```python
from deepbiop.core import KmerEncoder

encoder = KmerEncoder(k: int)
```

**Parameters:**
- `k` (int): K-mer size (typically 3-7)

**Methods:**
- `encode(sequence: bytes) -> numpy.ndarray`: Encode sequence to k-mer indices

**Returns:**
Array of k-mer indices (length = seq_len - k + 1)

**Example:**
```python
encoder = KmerEncoder(k=3)

sequence = b"ACGTACGT"
kmers = encoder.encode(sequence)
# Encodes: ACG, CGT, GTA, TAC, ACG, CGT
# Shape: (6,) for sequence of length 8 with k=3
```

---

### Composition

#### `Compose`

Compose multiple transforms sequentially.

```python
from deepbiop.transforms import Compose

pipeline = Compose(transforms: list)
```

**Parameters:**
- `transforms` (list): List of transform objects with `.augment()`, `.__call__()`, or `.apply()` methods

**Methods:**
- `__call__(record: dict) -> dict`: Apply all transforms sequentially
- `filter(record: dict) -> bool`: Apply all filter methods (if available)

**Example:**
```python
from deepbiop import Compose, fq

pipeline = Compose([
    fq.ReverseComplement(),
    fq.Mutator(mutation_rate=0.05),
    fq.OneHotEncoder(encoding_type="dna")
])

for record in dataset:
    transformed = pipeline(record)
    # transformed['sequence'] is reverse-complemented, mutated, and one-hot encoded
```

---

#### `FilterCompose`

Compose multiple filters with AND logic.

```python
from deepbiop.transforms import FilterCompose

filters = FilterCompose(filters: list)
```

**Parameters:**
- `filters` (list): List of filter objects with `.filter()` or `.passes()` methods

**Methods:**
- `filter(record: dict) -> bool`: Returns True only if ALL filters pass
- `__call__(record: dict) -> bool`: Alias for `.filter()`

**Example:**
```python
from deepbiop import FilterCompose, fq

filters = FilterCompose([
    fq.QualityFilter(min_mean_quality=30.0),
    fq.LengthFilter(min_length=100, max_length=300)
])

for record in dataset:
    if filters.filter(record):
        # Passed both quality and length filters
        process(record)
```

---

#### `TransformDataset`

Wrapper to apply transforms and filters during iteration.

```python
from deepbiop.transforms import TransformDataset

dataset = TransformDataset(
    dataset,
    transform=None,
    filter_fn=None
)
```

**Parameters:**
- `dataset`: Base dataset (iterable)
- `transform`: Transform or Compose object to apply
- `filter_fn`: Filter or FilterCompose object to apply

**Returns:**
Iterator that yields filtered and transformed records

**Example:**
```python
from deepbiop import TransformDataset, Compose, FilterCompose, fq

base_dataset = fq.FastqStreamDataset("reads.fastq.gz")

processed_dataset = TransformDataset(
    base_dataset,
    transform=Compose([
        fq.Mutator(mutation_rate=0.1),
        fq.OneHotEncoder()
    ]),
    filter_fn=FilterCompose([
        fq.QualityFilter(min_mean_quality=25.0),
        fq.LengthFilter(min_length=50)
    ])
)

for record in processed_dataset:
    # Record has been filtered and transformed
    train_model(record)
```

---

## PyTorch Integration

### BiologicalDataModule

PyTorch Lightning DataModule for biological data.

```python
from deepbiop.lightning import BiologicalDataModule

dm = BiologicalDataModule(
    train_path: str | None = None,
    val_path: str | None = None,
    test_path: str | None = None,
    batch_size: int = 32,
    num_workers: int = 0,
    file_type: str | None = None
)
```

**Parameters:**
- `train_path` (str, optional): Path to training data file
- `val_path` (str, optional): Path to validation data file
- `test_path` (str, optional): Path to test data file
- `batch_size` (int, default: 32): Batch size for DataLoaders
- `num_workers` (int, default: 0): Number of worker processes
- `file_type` (str, optional): File type ("fastq", "fasta", "bam"), auto-detected if None

**Methods:**
- `setup(stage: str | None = None)`: Create datasets ("fit", "test", or None for all)
- `train_dataloader() -> DataLoader`: Get training DataLoader
- `val_dataloader() -> DataLoader`: Get validation DataLoader
- `test_dataloader() -> DataLoader`: Get test DataLoader

**Example:**
```python
from deepbiop.lightning import BiologicalDataModule
import pytorch_lightning as pl

# Create data module
dm = BiologicalDataModule(
    train_path="train.fastq.gz",
    val_path="val.fastq.gz",
    batch_size=64,
    num_workers=4
)

# Use with Lightning Trainer
trainer = pl.Trainer(max_epochs=10, accelerator='gpu')
trainer.fit(model, dm)
```

---

## Core Utilities

### K-mer Generation

```python
from deepbiop.core import generate_kmers, generate_kmers_table

# Generate k-mers from sequence
kmers = generate_kmers(sequence: bytes, k: int) -> list[str]

# Generate all possible k-mers
kmer_table = generate_kmers_table(k: int, alphabet: str = "ACGT") -> list[str]
```

**Example:**
```python
from deepbiop.core import generate_kmers, generate_kmers_table

# All possible 3-mers
all_3mers = generate_kmers_table(k=3)
# ['AAA', 'AAC', 'AAG', 'AAT', 'ACA', ...]

# K-mers from sequence
sequence = b"ACGTACGT"
kmers = generate_kmers(sequence, k=3)
# ['ACG', 'CGT', 'GTA', 'TAC', 'ACG', 'CGT']
```

---

### Sequence Utilities

```python
from deepbiop.core import reverse_complement, normalize_seq

# Reverse complement
rc_seq = reverse_complement(sequence: bytes) -> bytes

# Normalize sequence (uppercase, validate)
norm_seq = normalize_seq(sequence: bytes) -> bytes
```

**Example:**
```python
from deepbiop.core import reverse_complement, normalize_seq

seq = b"acgt"
normalized = normalize_seq(seq)  # b"ACGT"
rc = reverse_complement(normalized)  # b"ACGT" (palindrome)
```

---

## Type Signatures

All public APIs include type hints. For complete type information, see the `.pyi` stub files:

- `deepbiop/fq.pyi` - FASTQ module types
- `deepbiop/fa.pyi` - FASTA module types
- `deepbiop/bam.pyi` - BAM module types
- `deepbiop/core.pyi` - Core utilities types
- `deepbiop/utils.pyi` - Utility module types

---

## Performance Characteristics

| Operation | Time Complexity | Memory |
|-----------|----------------|--------|
| `FastqStreamDataset` iteration | O(n) | O(1) constant |
| `FastqDataset` indexing | O(1) amortized | O(n) for index |
| `QualityFilter.passes()` | O(m) | O(1) |
| `LengthFilter.passes()` | O(1) | O(1) |
| `Mutator.augment()` | O(m) | O(m) |
| `OneHotEncoder` | O(m) | O(4m) |
| `KmerEncoder` | O(m) | O(m-k+1) |

Where:
- n = number of records in file
- m = sequence length

---

## See Also

- [Quick Start Guide](quickstart.md) - Get started in 5 minutes
- [Examples](../examples/) - Jupyter notebooks with complete workflows
- [Troubleshooting](troubleshooting.md) - Common errors and solutions
- [Performance Guide](performance.md) - Optimization tips and benchmarks

---

**Last Updated**: 2025-11-07
**Version**: 0.1.16
