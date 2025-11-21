"""Transform composition utilities for biological data.

This module provides utilities for composing and chaining transformations
on biological sequence data.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

    from deepbiop.core import Record


class Transform(ABC):
    """Abstract base class for all data transformations.

    Transforms operate on Record objects and return modified Record objects.
    All transforms must be:
    - Stateless (except for seeded random operations)
    - Picklable (for multiprocessing)
    - Deterministic when seed is provided

    Parameters
    ----------
    seed : int | None
        Random seed for reproducibility. If None, operations are non-deterministic.

    Examples:
    --------
    >>> class MyTransform(Transform):
    ...     def __call__(self, record):
    ...         # Modify record
    ...         return record
    """

    def __init__(self, seed: int | None = None, **kwargs: Any) -> None:
        """Initialize transform.

        Parameters
        ----------
        seed : int | None
            Random seed for reproducibility
        **kwargs : Any
            Transform-specific parameters
        """
        self.seed = seed
        self._rng: np.random.RandomState | None = None
        if seed is not None:
            import numpy as np

            self._rng = np.random.RandomState(seed)

    @abstractmethod
    def __call__(self, record: Record | dict[str, Any]) -> Record | dict[str, Any]:
        """Apply transform to a record.

        Parameters
        ----------
        record : Record | dict[str, Any]
            Input record

        Returns:
        -------
        Record | dict[str, Any]
            Transformed record

        Notes:
        -----
        Implementations must NOT modify the input record. Create a copy
        if modifications are needed.
        """
        ...

    def save_state(self) -> dict[str, Any]:
        """Save internal state for reproducibility.

        Returns:
        -------
        dict[str, Any]
            State dictionary that can be passed to load_state()
        """
        return {
            "seed": self.seed,
            "rng_state": self._rng.get_state() if self._rng else None,
        }

    def load_state(self, state: dict[str, Any]) -> None:
        """Restore internal state.

        Parameters
        ----------
        state : dict[str, Any]
            State dictionary from save_state()
        """
        import numpy as np

        self.seed = state["seed"]
        if state["rng_state"] is not None:
            self._rng = np.random.RandomState()
            self._rng.set_state(state["rng_state"])


class Compose:
    """Compose multiple transforms together.

    This class allows chaining multiple transformations that will be
    applied sequentially to data records.

    Parameters
    ----------
        transforms: List of transform objects with .augment() or .filter() methods

    Example:
        >>> from deepbiop.fq import ReverseComplement, Mutator
        >>> from deepbiop.transforms import Compose
        >>>
        >>> # Create composed transform
        >>> transform = Compose([ReverseComplement(), Mutator(mutation_rate=0.1)])
        >>>
        >>> # Apply to a record
        >>> record = {"id": "seq1", "sequence": b"ACGT", "quality": None}
        >>> transformed = transform(record)
    """

    def __init__(self, transforms: list[Any]):
        """Initialize Compose with a list of transforms.

        Args:
            transforms: List of transform objects
        """
        self.transforms = transforms

    def __call__(self, record: dict[str, Any]) -> dict[str, Any]:
        """Apply all transforms sequentially to a record.

        Args:
            record: Input record dict

        Returns:
        -------
            Transformed record dict
        """
        result = record
        for transform in self.transforms:
            # Try augment method first (for augmentation transforms)
            if hasattr(transform, "augment"):
                result = transform.augment(result)
            # Try __call__ method (for encoder transforms)
            elif callable(transform) and not isinstance(transform, type):
                result = transform(result)
            # Try apply method (generic)
            elif hasattr(transform, "apply"):
                result = transform.apply(result)
            else:
                msg = (
                    f"Transform {type(transform).__name__} must have "
                    f"augment(), __call__(), or apply() method"
                )
                raise TypeError(msg)
        return result

    def filter(self, record: dict[str, Any]) -> bool:
        """Apply filter transforms.

        Returns False if any filter rejects the record.

        Args:
            record: Input record dict

        Returns:
        -------
            True if record passes all filters, False otherwise
        """
        for transform in self.transforms:
            if hasattr(transform, "filter"):
                if not transform.filter(record):
                    return False
        return True

    def __repr__(self) -> str:
        """String representation."""
        transform_names = [type(t).__name__ for t in self.transforms]
        return f"Compose([{', '.join(transform_names)}])"


class FilterCompose:
    """Compose multiple filter transforms with AND logic.

    All filters must pass for a record to be accepted.

    Parameters
    ----------
        filters: List of filter objects with .filter() method

    Example:
        >>> from deepbiop.fq import QualityFilter, LengthFilter
        >>> from deepbiop.transforms import FilterCompose
        >>>
        >>> # Create composed filter
        >>> filter_chain = FilterCompose(
        ...     [
        ...         QualityFilter(min_quality=30.0),
        ...         LengthFilter(min_length=50, max_length=500),
        ...     ]
        ... )
        >>>
        >>> # Apply to a record
        >>> record = {"id": "seq1", "sequence": b"ACGT" * 20, "quality": b"I" * 80}
        >>> passes = filter_chain.filter(record)
    """

    def __init__(self, filters: list[Any]):
        """Initialize FilterCompose with a list of filters.

        Args:
            filters: List of filter objects
        """
        self.filters = filters

    def filter(self, record: dict[str, Any]) -> bool:
        """Apply all filters with AND logic.

        Args:
            record: Input record dict

        Returns:
        -------
            True if record passes all filters, False otherwise
        """
        return all(filt.filter(record) for filt in self.filters)

    def __call__(self, record: dict[str, Any]) -> bool:
        """Alias for filter() method."""
        return self.filter(record)

    def __repr__(self) -> str:
        """String representation."""
        filter_names = [type(f).__name__ for f in self.filters]
        return f"FilterCompose([{', '.join(filter_names)}])"


class TransformDataset:
    """Wrapper to apply transforms and extract targets from a dataset during iteration.

    This allows lazy application of transforms and target extraction as records
    are streamed from the underlying dataset, enabling supervised learning.

    Parameters
    ----------
        dataset: Underlying dataset (must be iterable)
        transform: Transform or Compose object to apply (encodes sequences)
        target_fn: Function or TargetExtractor to extract targets
        filter_fn: Optional filter function/object
        return_dict: If True, return dict samples; if False, return (features, target) tuples

    Example:
        >>> from deepbiop.fq import FastqStreamDataset, OneHotEncoder
        >>> from deepbiop.transforms import TransformDataset
        >>> from deepbiop.targets import TargetExtractor
        >>>
        >>> # Create base dataset
        >>> dataset = FastqStreamDataset("data.fastq")
        >>>
        >>> # Wrap with transforms and target extraction
        >>> transformed_dataset = TransformDataset(
        ...     dataset,
        ...     transform=OneHotEncoder(),
        ...     target_fn=TargetExtractor.from_quality("mean"),
        ...     return_dict=False,  # Return (features, target) tuples
        ... )
        >>>
        >>> # Iterate with transforms applied
        >>> for features, target in transformed_dataset:
        ...     # Ready for PyTorch training
        ...     pass
    """

    def __init__(
        self,
        dataset: Any,
        transform: Any = None,
        target_fn: Any = None,
        filter_fn: Any = None,
        *,
        return_dict: bool = True,
    ):
        """Initialize TransformDataset.

        Args:
            dataset: Underlying dataset
            transform: Transform to apply (encoder)
            target_fn: Function to extract targets
            filter_fn: Optional filter
            return_dict: If True, return dict; if False, return tuple
        """
        self.dataset = dataset
        self.transform = transform
        self.target_fn = target_fn
        self.filter_fn = filter_fn
        self.return_dict = return_dict

    def __iter__(self):
        """Iterate with transforms, filters, and target extraction applied.

        Yields:
        ------
            Transformed records (dict or tuple based on return_dict parameter)
        """
        for record in self.dataset:
            # Apply filter first
            if self.filter_fn is not None:
                if hasattr(self.filter_fn, "filter"):
                    if not self.filter_fn.filter(record):
                        continue
                elif callable(self.filter_fn):
                    if not self.filter_fn(record):
                        continue

            # Keep original record for target extraction (before transformation)
            original_record = record.copy() if isinstance(record, dict) else record

            # Apply transform to encode sequences
            if self.transform is not None:
                if hasattr(self.transform, "augment"):
                    record = self.transform.augment(record)
                elif hasattr(self.transform, "encode"):
                    # Encoder transforms: encode sequence and store in "features"
                    encoded = self.transform.encode(record["sequence"])
                    record["features"] = encoded
                elif callable(self.transform):
                    record = self.transform(record)

            # Extract target if configured
            if self.target_fn is not None:
                if callable(self.target_fn):
                    target = self.target_fn(original_record)
                else:
                    msg = f"target_fn must be callable, got {type(self.target_fn)}"
                    raise TypeError(msg)

                # Add target to record
                if isinstance(record, dict):
                    record["target"] = target
                else:
                    # Handle non-dict records
                    record = {"data": record, "target": target}

            # Return in appropriate format
            if not self.return_dict and self.target_fn is not None:
                # Return (features, target) tuple for PyTorch compatibility
                features = record.get("features", record.get("sequence"))
                target = record.get("target")
                yield (features, target)
            else:
                # Return dict (default or when no target)
                yield record

    def __len__(self):
        """Return dataset length if available.

        Note: Length may not be accurate if filtering is applied.
        """
        if hasattr(self.dataset, "__len__"):
            return len(self.dataset)
        return 0

    def __getitem__(self, idx):
        """Get item by index (for PyTorch DataLoader compatibility).

        Args:
            idx (int): Index of item to retrieve

        Returns:
            Transformed record at index idx
        """
        if not hasattr(self.dataset, "__getitem__"):
            msg = (
                f"Underlying dataset {type(self.dataset).__name__} does not support indexing. "
                f"Use iteration instead or ensure dataset has __getitem__ method."
            )
            raise TypeError(msg)

        # Get record from underlying dataset
        record = self.dataset[idx]

        # Keep original record for target extraction (before transformation)
        original_record = record.copy() if isinstance(record, dict) else record

        # Apply transform to encode sequences
        if self.transform is not None:
            if hasattr(self.transform, "augment"):
                record = self.transform.augment(record)
            elif hasattr(self.transform, "encode"):
                # Encoder transforms: encode sequence and store in "features"
                encoded = self.transform.encode(record["sequence"])
                record["features"] = encoded
            elif callable(self.transform):
                record = self.transform(record)

        # Extract target if configured
        if self.target_fn is not None:
            if callable(self.target_fn):
                target = self.target_fn(original_record)
            else:
                msg = f"target_fn must be callable, got {type(self.target_fn)}"
                raise TypeError(msg)

            # Add target to record
            if isinstance(record, dict):
                record["target"] = target
            else:
                # Handle non-dict records
                record = {"data": record, "target": target}

        # Return in appropriate format
        if not self.return_dict and self.target_fn is not None:
            # Return (features, target) tuple for PyTorch compatibility
            features = record.get("features", record.get("sequence"))
            target = record.get("target")
            return (features, target)
        else:
            # Return dict (default or when no target)
            return record

    def __repr__(self) -> str:
        """String representation."""
        transform_name = type(self.transform).__name__ if self.transform else "None"
        filter_name = type(self.filter_fn).__name__ if self.filter_fn else "None"
        return (
            f"TransformDataset(dataset={type(self.dataset).__name__}, "
            f"transform={transform_name}, filter={filter_name})"
        )


__all__ = ["Compose", "FilterCompose", "Transform", "TransformDataset"]
