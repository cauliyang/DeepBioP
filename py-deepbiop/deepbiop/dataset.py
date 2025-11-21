"""Dataset abstractions for biological sequence data."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

    from deepbiop.core import Record


class Dataset(ABC):
    """Abstract base class for all biological sequence datasets.

    All dataset implementations must inherit from this class and implement
    the required abstract methods. This provides a consistent interface
    compatible with PyTorch's Dataset protocol.

    Examples:
    --------
    >>> class MyDataset(Dataset):
    ...     def __len__(self):
    ...         return 100
    ...
    ...     def __getitem__(self, idx):
    ...         return Record(id=b"@r", sequence=b"ACGT", quality=b"IIII")
    ...
    ...     def __iter__(self):
    ...         for i in range(100):
    ...             yield Record(id=b"@r", sequence=b"ACGT", quality=b"IIII")
    """

    @abstractmethod
    def __len__(self) -> int:
        """Return the total number of records in the dataset.

        Returns:
        -------
        int
            Number of records

        Notes:
        -----
        For streaming datasets without indexing, this may return the total
        number of records after a full scan, or raise NotImplementedError
        if the count is not available.
        """
        ...

    @abstractmethod
    def __getitem__(self, idx: int) -> Record:
        """Get record at specified index.

        Parameters
        ----------
        idx : int
            Zero-based record index

        Returns:
        -------
        Record
            Record at the specified index

        Raises:
        ------
        IndexError
            If index is out of bounds
        NotImplementedError
            If random access is not supported (streaming mode)
        """
        ...

    @abstractmethod
    def __iter__(self) -> Iterator[Record]:
        """Iterate over all records in the dataset.

        Yields:
        ------
        Record
            Next record in the dataset

        Notes:
        -----
        This is the primary iteration method and must be implemented.
        Iteration should be memory-efficient for large files.
        """
        ...

    def __repr__(self) -> str:
        """Return string representation of dataset."""
        return f"{self.__class__.__name__}(len={len(self)})"
