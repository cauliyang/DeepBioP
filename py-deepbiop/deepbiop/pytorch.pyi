"""PyTorch-style Python API for biological sequence data loading.

This module provides PyTorch-compatible Dataset, DataLoader, and transform classes
for loading and preprocessing FASTQ/FASTA files, enabling researchers to use familiar
PyTorch patterns with biological sequence data.

Example:
    >>> from deepbiop.pytorch import Dataset, DataLoader, OneHotEncoder
    >>> # Create dataset with encoding
    >>> dataset = Dataset("data.fastq", transform=OneHotEncoder())
    >>> # Create data loader
    >>> loader = DataLoader(dataset, batch_size=32, shuffle=True)
    >>> # Iterate through batches
    >>> for batch in loader:
    ...     sequences = batch["sequences"]  # NumPy array
    ...     # ... training logic
"""

from collections.abc import Callable
from typing import Any

__all__ = [
    "Compose",
    "DataLoader",
    "DataLoaderIterator",
    "Dataset",
    "DatasetIterator",
    "IntegerEncoder",
    "KmerEncoder",
    "Mutator",
    "OneHotEncoder",
    "ReverseComplement",
    "Sampler",
    "default_collate",
    "is_cache_valid",
    "load_cache",
    "save_cache",
]

# Type aliases
Sample = dict[str, Any]
Transform = Callable[[Sample], Sample]
CollateFunction = Callable[[list[Sample]], Any]

class Dataset:
    """PyTorch-compatible Dataset for biological sequence files.

    Provides indexed access to FASTQ/FASTA files with optional transformations.
    Compatible with PyTorch DataLoader and other ML frameworks.

    Args:
        file_paths: Path to FASTQ/FASTA file (string) or list of paths
        sequence_type: Type of sequences - "dna", "rna", or "protein" (default: "dna")
        transform: Optional transformation pipeline to apply to samples
        cache_dir: Optional directory for caching processed data
        lazy: Load sequences on-demand if True, preload if False (default: True)

    Raises:
    ------
        FileNotFoundError: If the specified file doesn't exist
        ValueError: If file format is invalid or unsupported

    Example:
        >>> dataset = Dataset("data.fastq")
        >>> print(len(dataset))  # Number of sequences
        >>> sample = dataset[0]  # Get first sample
        >>> print(sample["sequence"])  # Access sequence
    """

    def __init__(
        self,
        file_paths: str | list[str],
        *,
        sequence_type: str = "dna",
        transform: Transform | None = None,
        cache_dir: str | None = None,
        lazy: bool = True,
    ) -> None: ...
    def __len__(self) -> int:
        """Return the total number of sequences in the dataset."""
    def __getitem__(self, idx: int) -> Sample:
        """Get sample at index idx.

        Args:
            idx: Sample index (0 to len(dataset)-1)

        Returns:
        -------
            Dictionary with 'sequence' (bytes) and 'quality' (bytes) keys

        Raises:
        ------
            IndexError: If idx is out of bounds
        """
    def __iter__(self) -> DatasetIterator:
        """Return iterator over dataset samples."""
    def __repr__(self) -> str:
        """Return string representation of dataset."""
    def summary(self) -> dict[str, Any]:
        """Get dataset summary statistics.

        Returns:
        -------
            Dictionary containing:
                - num_samples: Total number of sequences
                - length_stats: Min, max, mean, median sequence lengths
                - memory_footprint: Estimated memory usage in bytes
        """
    def validate(self) -> dict[str, Any]:
        """Validate dataset integrity.

        Returns:
        -------
            Dictionary containing:
                - is_valid: Boolean indicating validity
                - warnings: List of warning messages
                - errors: List of error messages
        """

class DatasetIterator:
    """Iterator for Dataset class."""

    def __iter__(self) -> DatasetIterator: ...
    def __next__(self) -> Sample: ...

class DataLoader:
    """PyTorch-compatible DataLoader for batching and shuffling.

    Wraps a Dataset and provides efficient batch iteration with optional
    shuffling and custom collate functions.

    Args:
        dataset: Dataset instance to load from
        batch_size: Number of samples per batch (default: 1)
        shuffle: Whether to shuffle data every epoch (default: False)
        collate_fn: Function to collate samples into batches (default: default_collate)
        drop_last: Drop last incomplete batch if True (default: False)
        num_workers: Number of worker processes (currently unused, reserved for future)

    Example:
        >>> dataset = Dataset("data.fastq")
        >>> loader = DataLoader(dataset, batch_size=32, shuffle=True)
        >>> for batch in loader:
        ...     # Process batch
        ...     pass
    """

    def __init__(
        self,
        dataset: Dataset,
        *,
        batch_size: int = 1,
        shuffle: bool = False,
        collate_fn: CollateFunction | None = None,
        drop_last: bool = False,
        num_workers: int = 0,
    ) -> None: ...
    def __len__(self) -> int:
        """Return number of batches in the DataLoader."""
    def __iter__(self) -> DataLoaderIterator:
        """Return iterator over batches."""
    def __repr__(self) -> str:
        """Return string representation of DataLoader."""

class DataLoaderIterator:
    """Iterator for DataLoader class."""

    def __iter__(self) -> DataLoaderIterator: ...
    def __next__(self) -> list[Sample]: ...

class OneHotEncoder:
    """One-hot encoding transform for DNA/RNA/protein sequences.

    Encodes biological sequences as one-hot vectors suitable for CNNs and RNNs.
    For DNA: 4 channels (A, C, G, T)
    For RNA: 4 channels (A, C, G, U)
    For Protein: 20 channels (standard amino acids)

    Args:
        encoding_type: Type of encoding - "dna", "rna", or "protein"
        unknown_strategy: How to handle unknown bases - "skip", "zero", or "error"

    Returns:
    -------
        Transformed sample with sequence as NumPy array of shape (seq_len, num_channels)
        with dtype float32

    Example:
        >>> encoder = OneHotEncoder(encoding_type="dna", unknown_strategy="skip")
        >>> sample = {"sequence": b"ACGT"}
        >>> transformed = encoder(sample)
        >>> print(transformed["sequence"].shape)  # (4, 4)
    """

    def __init__(
        self, *, encoding_type: str = "dna", unknown_strategy: str = "skip"
    ) -> None: ...
    def __call__(self, sample: Sample) -> Sample:
        """Apply one-hot encoding to sample."""
    def __repr__(self) -> str: ...

class IntegerEncoder:
    """Integer encoding transform for DNA/RNA/protein sequences.

    Encodes biological sequences as integer arrays suitable for embeddings
    and transformer models.

    Args:
        encoding_type: Type of encoding - "dna", "rna", or "protein"
        unknown_strategy: How to handle unknown bases - "skip", "zero", or "error"

    Returns:
    -------
        Transformed sample with sequence as NumPy array of dtype int64

    Example:
        >>> encoder = IntegerEncoder(encoding_type="dna")
        >>> sample = {"sequence": b"ACGT"}
        >>> transformed = encoder(sample)
        >>> print(transformed["sequence"])  # array([0, 1, 2, 3])
    """

    def __init__(
        self, *, encoding_type: str = "dna", unknown_strategy: str = "skip"
    ) -> None: ...
    def __call__(self, sample: Sample) -> Sample:
        """Apply integer encoding to sample."""
    def __repr__(self) -> str: ...

class KmerEncoder:
    """K-mer encoding transform for DNA/RNA/protein sequences.

    Encodes sequences as k-mer frequency vectors suitable for feature-based
    models (Random Forest, SVM, XGBoost).

    Args:
        k: Length of k-mers (default: 3)
        canonical: Use canonical k-mers (consider reverse complements as same) (default: False)
        encoding_type: Type of encoding - "dna", "rna", or "protein"
        unknown_strategy: How to handle unknown bases - "skip" or "error"

    Returns:
    -------
        Transformed sample with sequence as NumPy array of k-mer frequencies

    Example:
        >>> encoder = KmerEncoder(k=3, canonical=False, encoding_type="dna")
        >>> sample = {"sequence": b"ACGTACGT"}
        >>> transformed = encoder(sample)
    """

    def __init__(
        self,
        *,
        k: int = 3,
        canonical: bool = False,
        encoding_type: str = "dna",
        unknown_strategy: str = "skip",
    ) -> None: ...
    def __call__(self, sample: Sample) -> Sample:
        """Apply k-mer encoding to sample."""
    def __repr__(self) -> str: ...

class Compose:
    """Compose multiple transforms into a single transform.

    Applies a sequence of transformations in order.

    Args:
        transforms: List of transform callables to apply in sequence

    Example:
        >>> transform = Compose(
        ...     [ReverseComplement(probability=0.5), OneHotEncoder(encoding_type="dna")]
        ... )
        >>> transformed = transform(sample)
    """

    def __init__(self, transforms: list[Transform]) -> None: ...
    def __call__(self, sample: Sample) -> Sample:
        """Apply all transforms in sequence."""
    def __repr__(self) -> str: ...

class ReverseComplement:
    """Reverse complement augmentation for DNA/RNA sequences.

    Randomly applies reverse complement transformation for orientation-invariant
    training data.

    Args:
        probability: Probability of applying reverse complement (0.0 to 1.0)
        seed: Optional random seed for reproducibility

    Example:
        >>> augment = ReverseComplement(probability=0.5, seed=42)
        >>> sample = {"sequence": b"ACGT"}
        >>> augmented = augment(sample)  # May return original or reverse complement
    """

    def __init__(
        self, *, probability: float = 0.5, seed: int | None = None
    ) -> None: ...
    def __call__(self, sample: Sample) -> Sample:
        """Apply reverse complement with specified probability."""
    def __repr__(self) -> str: ...

class Mutator:
    """Random mutation augmentation for sequences.

    Simulates SNPs and sequencing errors by randomly mutating bases.

    Args:
        mutation_rate: Probability of mutating each base (0.0 to 1.0)
        seed: Optional random seed for reproducibility

    Example:
        >>> augment = Mutator(mutation_rate=0.01, seed=42)
        >>> sample = {"sequence": b"ACGTACGT"}
        >>> mutated = augment(sample)  # Randomly mutated sequence
    """

    def __init__(
        self, *, mutation_rate: float = 0.01, seed: int | None = None
    ) -> None: ...
    def __call__(self, sample: Sample) -> Sample:
        """Apply random mutations to sequence."""
    def __repr__(self) -> str: ...

class Sampler:
    """Subsequence sampling augmentation.

    Extracts random or fixed-position windows from sequences.

    Args:
        length: Length of subsequence to extract
        mode: Sampling mode - "random", "start", "center", or "end"
        seed: Optional random seed for reproducibility

    Example:
        >>> sampler = Sampler(length=100, mode="random", seed=42)
        >>> sample = {"sequence": b"ACGT" * 50}  # 200bp sequence
        >>> windowed = sampler(sample)  # 100bp random window
    """

    def __init__(
        self, *, length: int, mode: str = "random", seed: int | None = None
    ) -> None: ...
    def __call__(self, sample: Sample) -> Sample:
        """Extract subsequence from sample."""
    def __repr__(self) -> str: ...

def default_collate(samples: list[Sample]) -> list[Sample]:
    """Default collate function for DataLoader.

    Currently returns samples as-is. Future implementations may provide
    intelligent batching (e.g., padding sequences to same length).

    Args:
        samples: List of sample dictionaries

    Returns:
    -------
        List of samples (currently unchanged)

    Example:
        >>> samples = [{"sequence": b"ACGT"}, {"sequence": b"TTGG"}]
        >>> batch = default_collate(samples)
    """

def save_cache(
    data: Any, cache_path: str, metadata: dict[str, Any] | None = None
) -> None:
    """Save processed data to cache file.

    Args:
        data: Data to cache (must be pickle-able)
        cache_path: Path to cache file
        metadata: Optional metadata dictionary (e.g., version, timestamp)

    Raises:
    ------
        IOError: If cache file cannot be written

    Example:
        >>> data = {"sequences": [...], "labels": [...]}
        >>> save_cache(data, "cache.pkl", metadata={"version": "1.0"})
    """

def load_cache(cache_path: str) -> tuple[Any, dict[str, Any] | None]:
    """Load processed data from cache file.

    Args:
        cache_path: Path to cache file

    Returns:
    -------
        Tuple of (data, metadata)

    Raises:
    ------
        FileNotFoundError: If cache file doesn't exist
        IOError: If cache file is corrupted

    Example:
        >>> data, metadata = load_cache("cache.pkl")
        >>> print(metadata["version"])
    """

def is_cache_valid(
    cache_path: str, source_files: list[str], max_age_seconds: float | None = None
) -> bool:
    """Check if cache is valid and up-to-date.

    Args:
        cache_path: Path to cache file
        source_files: List of source file paths to check against
        max_age_seconds: Optional maximum cache age in seconds

    Returns:
    -------
        True if cache exists and is newer than all source files (and within max_age if specified)

    Example:
        >>> if is_cache_valid("cache.pkl", ["data.fastq"], max_age_seconds=3600):
        ...     data, _ = load_cache("cache.pkl")
        ... else:
        ...     # Regenerate cache
        ...     pass
    """
