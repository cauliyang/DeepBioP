# API Contracts: PyTorch-Style Python API

**Date**: 2025-11-03
**Feature**: PyTorch-Style Python API for Deep Learning
**Phase**: 1 - Design

This document defines the complete Python API surface with type annotations, method signatures, and behavioral contracts.

---

## Module Structure

```python
from deepbiop.pytorch import (
    # Core classes
    Dataset,
    DataLoader,

    # Transforms
    Compose,
    OneHotEncoder,
    IntegerEncoder,
    KmerEncoder,
    ReverseComplement,
    Mutator,
    Sampler,

    # Collate functions
    default_collate_fn,
    pad_collate_fn,
    truncate_collate_fn,

    # Cache
    Cache,
)
```

---

## Type Definitions

```python
from typing import (
    Union, Optional, List, Dict, Any, Callable,
    Iterator, Tuple, TypedDict, Protocol
)
import numpy as np
from numpy.typing import NDArray

# Sample types
Sequence = Union[bytes, NDArray[np.float32]]
Label = Union[int, float, str]

class Sample(TypedDict, total=False):
    """Single data point from dataset"""
    sequence: Sequence
    quality: Optional[bytes]
    label: Optional[Label]
    metadata: Optional[Dict[str, Any]]

class Batch(TypedDict, total=False):
    """Batched samples for model input"""
    sequences: NDArray[np.float32]  # shape: [batch_size, max_len, features]
    labels: Optional[NDArray]        # shape: [batch_size]
    quality: Optional[NDArray]       # shape: [batch_size, max_len]
    metadata: Optional[List[Dict]]   # length: batch_size
    lengths: Optional[NDArray[np.int32]]  # shape: [batch_size]

# Collate function type
CollateFn = Callable[[List[Sample]], Batch]

# Transform protocol
class Transform(Protocol):
    """Protocol for all transforms"""
    def __call__(self, sample: Sample) -> Sample: ...
    def __repr__(self) -> str: ...
```

---

## Class: Dataset

**Purpose**: Wraps biological sequence files for PyTorch-compatible data loading.

### Constructor

```python
class Dataset:
    def __init__(
        self,
        file_paths: Union[str, List[str]],
        *,
        sequence_type: str = "dna",
        transform: Optional[Transform] = None,
        cache_dir: Optional[str] = None,
        lazy: bool = True
    ) -> None:
        """
        Create a dataset from FASTQ/FASTA files.

        Args:
            file_paths: Path(s) to FASTQ/FASTA files
            sequence_type: Type of sequences ("dna", "rna", or "protein")
            transform: Optional transformation pipeline
            cache_dir: Directory for caching processed data
            lazy: If True, load sequences on-demand (default)

        Raises:
            FileNotFoundError: If file_paths don't exist
            ValueError: If files are invalid format or sequence_type unknown
            PermissionError: If cache_dir is not writable

        Examples:
            >>> dataset = Dataset("data.fastq")
            >>> dataset = Dataset(["file1.fq", "file2.fq"], sequence_type="rna")
            >>> dataset = Dataset("data.fq", transform=OneHotEncoder())
        """
```

### Methods

```python
def __len__(self) -> int:
    """
    Returns total number of sequences in dataset.

    Returns:
        Number of sequences

    Examples:
        >>> len(dataset)
        1000
    """

def __getitem__(self, idx: int) -> Sample:
    """
    Get sample at index idx (applies transform if set).

    Args:
        idx: Sample index (0 to len(dataset)-1)

    Returns:
        Sample dict with 'sequence' key (+ 'quality', 'label', 'metadata' if available)

    Raises:
        IndexError: If idx out of range
        IOError: If file read fails

    Examples:
        >>> sample = dataset[0]
        >>> sample['sequence']  # bytes or ndarray (if transform applied)
        b'ACGTACGT'
    """

def __iter__(self) -> Iterator[Sample]:
    """
    Iterate over all samples in dataset.

    Yields:
        Sample dicts

    Examples:
        >>> for sample in dataset:
        ...     print(sample['sequence'])
    """

def summary(self) -> Dict[str, Any]:
    """
    Get dataset statistics using NumPy functions.

    Returns:
        Dict with keys:
            - 'num_samples': int
            - 'sequence_lengths': {'min': int, 'max': int, 'mean': float, 'std': float}
            - 'sequence_type': str
            - 'has_quality': bool
            - 'has_labels': bool

    Examples:
        >>> dataset.summary()
        {'num_samples': 1000, 'sequence_lengths': {'min': 100, 'max': 150, ...}, ...}
    """

def __repr__(self) -> str:
    """
    Returns:
        Human-readable representation

    Examples:
        >>> dataset
        Dataset(num_samples=1000, sequence_type='dna', transform=OneHotEncoder())
    """
```

---

## Class: DataLoader

**Purpose**: Generates batches from Dataset with shuffling and parallel loading.

### Constructor

```python
class DataLoader:
    def __init__(
        self,
        dataset: Dataset,
        *,
        batch_size: int = 1,
        shuffle: bool = False,
        num_workers: int = 0,
        collate_fn: Optional[CollateFn] = None,
        drop_last: bool = False,
        seed: Optional[int] = None
    ) -> None:
        """
        Create a data loader for batching and iteration.

        Args:
            dataset: Source dataset
            batch_size: Samples per batch (must be > 0)
            shuffle: Randomize sample order each epoch
            num_workers: Parallel loading processes (0 = main process only)
            collate_fn: Function to merge samples into batch (default: default_collate_fn)
            drop_last: Drop incomplete final batch
            seed: Random seed for reproducible shuffling

        Raises:
            ValueError: If batch_size <= 0 or num_workers < 0

        Examples:
            >>> loader = DataLoader(dataset, batch_size=32, shuffle=True)
            >>> loader = DataLoader(dataset, batch_size=16, collate_fn=pad_collate_fn)
        """
```

### Methods

```python
def __iter__(self) -> Iterator[Batch]:
    """
    Iterate over batches.

    Yields:
        Batch dicts

    Examples:
        >>> for batch in loader:
        ...     sequences = batch['sequences']  # shape: [batch_size, max_len, features]
    """

def __len__(self) -> int:
    """
    Returns number of batches.

    Returns:
        Number of batches (depends on batch_size and drop_last)

    Examples:
        >>> len(loader)
        32  # 1000 samples / 32 batch_size = 31.25 → 32 (drop_last=False)
    """

def __repr__(self) -> str:
    """Returns human-readable representation"""
```

---

## Class: Compose

**Purpose**: Chains multiple transforms into a pipeline.

### Constructor & Methods

```python
class Compose:
    def __init__(self, transforms: List[Transform]) -> None:
        """
        Create transformation pipeline.

        Args:
            transforms: Ordered list of transforms to apply

        Raises:
            ValueError: If transforms is empty

        Examples:
            >>> pipeline = Compose([
            ...     Sampler(length=100, strategy="random"),
            ...     Mutator(mutation_rate=0.01),
            ...     OneHotEncoder()
            ... ])
        """

    def __call__(self, sample: Sample) -> Sample:
        """
        Apply all transforms in sequence.

        Args:
            sample: Input sample

        Returns:
            Transformed sample

        Examples:
            >>> transformed = pipeline(sample)
        """

    def __repr__(self) -> str:
        """Returns representation showing all transforms"""
```

---

## Class: OneHotEncoder

**Purpose**: Encodes sequences as one-hot arrays (delegates to `deepbiop.fq.OneHotEncoder`).

```python
class OneHotEncoder:
    def __init__(
        self,
        encoding_type: str = "dna",
        unknown_strategy: str = "skip"
    ) -> None:
        """
        Create one-hot encoder.

        Args:
            encoding_type: Sequence type ("dna", "rna", "protein")
            unknown_strategy: How to handle unknown bases ("skip", "zero", "error")

        Examples:
            >>> encoder = OneHotEncoder()
            >>> encoder = OneHotEncoder(encoding_type="rna", unknown_strategy="zero")
        """

    def __call__(self, sample: Sample) -> Sample:
        """
        Encode sequence to one-hot array.

        Returns:
            Sample with 'sequence' as ndarray of shape [length, alphabet_size]
                - DNA/RNA: alphabet_size = 4
                - Protein: alphabet_size = 20
        """
```

---

## Class: IntegerEncoder

**Purpose**: Encodes sequences as integer arrays (A=0, C=1, G=2, T/U=3).

```python
class IntegerEncoder:
    def __init__(self, encoding_type: str = "dna") -> None:
        """
        Create integer encoder.

        Args:
            encoding_type: Sequence type ("dna", "rna", "protein")

        Examples:
            >>> encoder = IntegerEncoder()
        """

    def __call__(self, sample: Sample) -> Sample:
        """
        Encode sequence to integer array.

        Returns:
            Sample with 'sequence' as ndarray of shape [length] with integer values
        """
```

---

## Class: KmerEncoder

**Purpose**: Encodes sequences as k-mer frequency vectors.

```python
class KmerEncoder:
    def __init__(
        self,
        k: int,
        canonical: bool = False,
        encoding_type: str = "dna"
    ) -> None:
        """
        Create k-mer encoder.

        Args:
            k: K-mer length
            canonical: Treat k-mer and reverse complement as same
            encoding_type: Sequence type ("dna", "rna", "protein")

        Raises:
            ValueError: If k <= 0

        Examples:
            >>> encoder = KmerEncoder(k=3)
            >>> encoder = KmerEncoder(k=5, canonical=True)
        """

    def __call__(self, sample: Sample) -> Sample:
        """
        Encode sequence to k-mer frequency vector.

        Returns:
            Sample with 'sequence' as ndarray of shape [num_possible_kmers]
                - num_possible_kmers = alphabet_size^k
        """
```

---

## Class: ReverseComplement

**Purpose**: Applies reverse complement transformation.

```python
class ReverseComplement:
    def __init__(self, is_rna: bool = False) -> None:
        """
        Create reverse complement transformer.

        Args:
            is_rna: If True, use RNA bases (U instead of T)

        Examples:
            >>> rc = ReverseComplement()
            >>> rc_rna = ReverseComplement(is_rna=True)
        """

    def __call__(self, sample: Sample) -> Sample:
        """
        Apply reverse complement to sequence.

        Returns:
            Sample with reversed and complemented sequence
        """
```

---

## Class: Mutator

**Purpose**: Applies random mutations to sequences.

```python
class Mutator:
    def __init__(
        self,
        mutation_rate: float,
        seed: Optional[int] = None,
        is_rna: bool = False
    ) -> None:
        """
        Create mutator.

        Args:
            mutation_rate: Probability of mutating each base (0.0 to 1.0)
            seed: Random seed for reproducibility
            is_rna: If True, use RNA bases

        Raises:
            ValueError: If mutation_rate not in [0.0, 1.0]

        Examples:
            >>> mutator = Mutator(mutation_rate=0.01, seed=42)
        """

    def __call__(self, sample: Sample) -> Sample:
        """
        Apply random mutations to sequence.

        Returns:
            Sample with mutated sequence
        """
```

---

## Class: Sampler

**Purpose**: Extracts subsequences from sequences.

```python
class Sampler:
    def __init__(
        self,
        length: int,
        strategy: str = "random",
        seed: Optional[int] = None
    ) -> None:
        """
        Create sequence sampler.

        Args:
            length: Length of subsequence to extract
            strategy: Sampling strategy ("start", "center", "end", "random")
            seed: Random seed for random strategy

        Raises:
            ValueError: If length <= 0 or strategy invalid

        Examples:
            >>> sampler = Sampler(length=100, strategy="random", seed=42)
            >>> sampler_center = Sampler(length=150, strategy="center")
        """

    def __call__(self, sample: Sample) -> Sample:
        """
        Extract subsequence from sequence.

        Returns:
            Sample with sampled sequence of specified length

        Raises:
            ValueError: If sequence shorter than requested length
        """
```

---

## Class: Cache

**Purpose**: Manages persistent storage of processed datasets.

```python
class Cache:
    @staticmethod
    def save(
        dataset: Dataset,
        cache_dir: str
    ) -> str:
        """
        Save dataset to cache (delegates to existing Parquet export).

        Args:
            dataset: Dataset to cache
            cache_dir: Directory for cache files

        Returns:
            Path to cached file

        Raises:
            PermissionError: If cache_dir not writable
            IOError: If export fails

        Examples:
            >>> cache_path = Cache.save(dataset, "/tmp/cache")
        """

    @staticmethod
    def load(
        cache_path: str
    ) -> Dataset:
        """
        Load dataset from cache.

        Args:
            cache_path: Path to cached file

        Returns:
            Loaded dataset

        Raises:
            FileNotFoundError: If cache_path doesn't exist
            ValueError: If cache invalid or corrupted

        Examples:
            >>> dataset = Cache.load("/tmp/cache/dataset.parquet")
        """

    @staticmethod
    def is_valid(
        cache_path: str,
        source_files: List[str]
    ) -> bool:
        """
        Check if cache is still valid (source files unchanged).

        Args:
            cache_path: Path to cached file
            source_files: Original source files

        Returns:
            True if cache valid, False otherwise

        Examples:
            >>> if Cache.is_valid(cache_path, ["data.fq"]):
            ...     dataset = Cache.load(cache_path)
        """
```

---

## Collate Functions

### default_collate_fn

```python
def default_collate_fn(batch: List[Sample]) -> Batch:
    """
    Default collation: pad sequences to max length, stack arrays.

    Uses NumPy for padding and stacking (delegates to existing utils if available).

    Args:
        batch: List of samples from Dataset.__getitem__()

    Returns:
        Batch dict with:
            - 'sequences': NDArray of shape [batch_size, max_len, features]
            - 'labels': NDArray of shape [batch_size] (if labels present)
            - 'quality': NDArray of shape [batch_size, max_len] (if quality present)
            - 'metadata': List of dicts (if metadata present)
            - 'lengths': NDArray of original lengths before padding

    Examples:
        >>> batch = default_collate_fn([sample1, sample2, sample3])
        >>> batch['sequences'].shape
        (3, 150, 4)  # 3 samples, max_len=150, 4 features (one-hot DNA)
    """
```

### pad_collate_fn

```python
def pad_collate_fn(
    padding_value: float = 0.0,
    pad_to: Optional[int] = None
) -> CollateFn:
    """
    Create custom padding collate function.

    Args:
        padding_value: Value to use for padding
        pad_to: Fixed length to pad to (if None, use max in batch)

    Returns:
        Collate function

    Examples:
        >>> loader = DataLoader(
        ...     dataset,
        ...     batch_size=32,
        ...     collate_fn=pad_collate_fn(padding_value=-1, pad_to=200)
        ... )
    """
```

### truncate_collate_fn

```python
def truncate_collate_fn(max_length: int) -> CollateFn:
    """
    Create truncating collate function.

    Args:
        max_length: Maximum sequence length (truncate longer sequences)

    Returns:
        Collate function

    Examples:
        >>> loader = DataLoader(
        ...     dataset,
        ...     batch_size=32,
        ...     collate_fn=truncate_collate_fn(max_length=150)
        ... )
    """
```

---

## Usage Example

```python
from deepbiop.pytorch import (
    Dataset, DataLoader, Compose,
    Sampler, Mutator, OneHotEncoder,
    default_collate_fn
)
import torch
import torch.nn as nn

# 1. Create dataset with transformation pipeline
transform = Compose([
    Sampler(length=100, strategy="random", seed=42),
    Mutator(mutation_rate=0.01, seed=42),
    OneHotEncoder(encoding_type="dna")
])

dataset = Dataset(
    "data/sequences.fastq",
    sequence_type="dna",
    transform=transform,
    cache_dir="/tmp/cache"
)

# 2. Create data loader
loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    collate_fn=default_collate_fn,
    seed=42
)

# 3. Use with PyTorch model
model = nn.Sequential(
    nn.Conv1d(4, 64, kernel_size=7),
    nn.ReLU(),
    nn.AdaptiveAvgPool1d(1),
    nn.Flatten(),
    nn.Linear(64, 2)
)

for batch in loader:
    # Zero-copy conversion from NumPy to PyTorch tensor
    sequences = torch.from_numpy(batch['sequences'])  # [batch_size, length, 4]
    sequences = sequences.permute(0, 2, 1)  # [batch_size, 4, length] for Conv1d

    outputs = model(sequences)
    # ... training logic
```

---

## Error Handling

All methods raise standard Python exceptions with descriptive messages:

```python
# File errors
FileNotFoundError("FASTQ file 'data.fq' not found. Check path and permissions.")

# Validation errors
ValueError("Invalid nucleotide 'N' at position 42 in sequence. Expected A, C, G, T for DNA.")
ValueError("batch_size must be positive, got 0")
ValueError("mutation_rate must be in [0.0, 1.0], got 1.5")

# Index errors
IndexError("Index 1000 out of range (dataset size: 500)")

# I/O errors
IOError("Failed to read sequence at index 123: file corrupted")
PermissionError("Cache directory '/tmp/cache' is not writable")
```

---

## Performance Contracts

- **SC-002**: Data loading overhead < 10% of total training time
- **SC-005**: Batch generation processes 10,000+ sequences/second
- **SC-006**: Dataset.summary() completes in < 1s per 10k sequences

All transforms release GIL for Rayon parallel operations (via `py.detach()` pattern from research.md).

---

## Compatibility

- **Python Version**: ≥ 3.10
- **NumPy Version**: ≥ 1.20
- **PyTorch Version**: ≥ 2.0 (optional, for tensor conversion)
- **Platform**: Linux, macOS, Windows

Zero-copy conversion to PyTorch tensors via `torch.from_numpy()`:

```python
# NumPy array from DeepBioP (shares memory)
sequences_np = batch['sequences']  # NDArray[float32]

# PyTorch tensor (zero-copy, shares memory with NumPy)
sequences_torch = torch.from_numpy(sequences_np)  # Tensor[float32]
```

---

## Next Artifact

- **quickstart.md**: Complete working example from installation to training
