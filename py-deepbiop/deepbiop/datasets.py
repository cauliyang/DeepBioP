"""
PyTorch-compatible dataset wrappers for biological sequence data.

This module provides simple wrappers to read FASTQ/FASTA files with a
standard PyTorch Dataset interface (returning individual samples).
"""

from typing import Any, Iterator


class FastqDataset:
    """
    Simple PyTorch-compatible FASTQ dataset that returns individual records.

    This implementation uses a simple streaming approach with optional caching
    for better performance with PyTorch DataLoader.

    Parameters
    ----------
        file_path: Path to FASTQ file (plain, gzipped, or bgzipped)

    Example:
        >>> from torch.utils.data import DataLoader
        >>> dataset = FastqDataset("data.fastq.gz")
        >>> loader = DataLoader(dataset, batch_size=32)
        >>> for batch in loader:
        ...     # Process batch of individual records
        ...     pass

    Note:
        For best performance with multi-worker DataLoader, the underlying
        Rust implementation handles file reading efficiently.
    """

    def __init__(self, file_path: str):
        """Initialize FastqDataset with file path."""
        from deepbiop.fq import FastqStreamDataset

        self.file_path = file_path

        # Use Rust FastqStreamDataset which provides efficient streaming
        self._rust_dataset = FastqStreamDataset(file_path)

        # Get total count by reading the file once
        # This is needed for __len__ and random access
        self._records_cache = list(self._read_all_records())
        self._total_records = len(self._records_cache)

    def _read_all_records(self) -> Iterator[dict[str, Any]]:
        """Read all records from file using Rust streaming dataset."""
        # Iterate through Rust dataset (already returns dicts)
        for record_dict in self._rust_dataset:
            # rust_record is already a dict with id, sequence, quality keys
            yield record_dict

    def __len__(self) -> int:
        """Return total number of records."""
        return self._total_records

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """
        Get individual record at index.

        Parameters
        ----------
            idx: Record index (0-based)

        Returns
        -------
            Record dict with keys: "id", "sequence", "quality"
        """
        if idx < 0 or idx >= self._total_records:
            raise IndexError(f"Index {idx} out of range [0, {self._total_records})")

        return self._records_cache[idx]

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """
        Iterate over all records.

        Yields
        ------
            Record dict with keys: "id", "sequence", "quality"
        """
        return iter(self._records_cache)

    def __repr__(self) -> str:
        """String representation."""
        return f"FastqDataset(file_path='{self.file_path}', total_records={self._total_records})"


__all__ = ["FastqDataset"]
