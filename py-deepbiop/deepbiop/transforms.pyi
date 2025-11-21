"""Transform composition utilities for biological sequence data.

Provides utilities for composing and chaining transformations in preprocessing
pipelines.
"""

from collections.abc import Callable, Iterator
from typing import Any

__all__ = ["Compose", "FilterCompose", "TransformDataset"]

Sample = dict[str, Any]
Transform = Callable[[Sample], Sample]
Filter = Callable[[Sample], bool]

class Compose:
    """Compose multiple transforms into a single transform.

    Applies a sequence of transformations in order, passing the output of each
    transform as input to the next.

    Args:
        transforms: List of transform callables to apply in sequence

    Example:
        >>> from deepbiop.transforms import Compose
        >>> from deepbiop.pytorch import ReverseComplement, OneHotEncoder
        >>> transform = Compose(
        ...     [ReverseComplement(probability=0.5), OneHotEncoder(encoding_type="dna")]
        ... )
        >>> sample = {"sequence": b"ACGT"}
        >>> transformed = transform(sample)
    """

    transforms: list[Transform]

    def __init__(self, transforms: list[Transform]) -> None: ...
    def __call__(self, sample: Sample) -> Sample:
        """Apply all transforms in sequence to the sample."""
    def __repr__(self) -> str: ...

class FilterCompose:
    """Compose multiple filters with AND logic.

    Applies filters in sequence, short-circuiting on first False result.

    Args:
        filters: List of filter callables (return True to keep sample)

    Example:
        >>> from deepbiop.transforms import FilterCompose
        >>> from deepbiop.fq import QualityFilter, LengthFilter
        >>> filter_fn = FilterCompose(
        ...     [
        ...         QualityFilter(min_quality=20),
        ...         LengthFilter(min_length=50, max_length=500),
        ...     ]
        ... )
        >>> sample = {"sequence": b"ACGT", "quality": b"IIII"}
        >>> if filter_fn(sample):
        ...     # Process sample
        ...     pass
    """

    filters: list[Filter]

    def __init__(self, filters: list[Filter]) -> None: ...
    def __call__(self, sample: Sample) -> bool:
        """Return True if all filters pass, False otherwise."""
    def __repr__(self) -> str: ...

class TransformDataset:
    """Wrapper dataset that applies transformations to another dataset.

    Lazily applies transforms on-the-fly during iteration or indexing.

    Args:
        dataset: Base dataset to wrap
        transform: Transform callable to apply to each sample
        filter_fn: Optional filter callable (samples that return False are skipped)

    Example:
        >>> from deepbiop.transforms import TransformDataset
        >>> from deepbiop.pytorch import Dataset, OneHotEncoder
        >>> base_dataset = Dataset("data.fastq")
        >>> transformed = TransformDataset(
        ...     base_dataset, transform=OneHotEncoder(encoding_type="dna")
        ... )
        >>> sample = transformed[0]  # Returns one-hot encoded sample
    """

    dataset: Any
    transform: Transform | None
    filter_fn: Filter | None

    def __init__(
        self,
        dataset: Any,
        transform: Transform | None = None,
        filter_fn: Filter | None = None,
    ) -> None: ...
    def __len__(self) -> int:
        """Return number of samples in dataset."""
    def __getitem__(self, idx: int) -> Sample:
        """Get sample at index, applying transforms and filters."""
    def __iter__(self) -> Iterator[Sample]:
        """Iterate over dataset, applying transforms and filters."""
    def __repr__(self) -> str: ...
