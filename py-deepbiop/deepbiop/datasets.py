"""PyTorch-compatible dataset wrappers for biological sequence data.

This module provides simple wrappers to read FASTQ/FASTA files with a
standard PyTorch Dataset interface (returning individual samples).
"""

from collections.abc import Iterator
from typing import Any


class FastqDataset:
    """Simple PyTorch-compatible FASTQ dataset that returns individual records.

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
        yield from self._rust_dataset

    def __len__(self) -> int:
        """Return total number of records."""
        return self._total_records

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get individual record at index.

        Args:
            idx (int): Record index (0-based)

        Returns:
            Record dict with keys: "id", "sequence", "quality"
        """
        if idx < 0 or idx >= self._total_records:
            msg = f"Index {idx} out of range [0, {self._total_records})"
            raise IndexError(msg)

        return self._records_cache[idx]

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over all records.

        Yields:
            Record dict with keys: "id", "sequence", "quality"
        """
        return iter(self._records_cache)

    def __repr__(self) -> str:
        """String representation."""
        return f"FastqDataset(file_path='{self.file_path}', total_records={self._total_records})"


class FastaDataset:
    """Simple PyTorch-compatible FASTA dataset that returns individual records.

    This implementation uses a simple streaming approach with optional caching
    for better performance with PyTorch DataLoader.

    Parameters
    ----------
        file_path: Path to FASTA file (plain, gzipped, or bgzipped)

    Example:
        >>> from torch.utils.data import DataLoader
        >>> dataset = FastaDataset("genome.fasta.gz")
        >>> loader = DataLoader(dataset, batch_size=32)
        >>> for batch in loader:
        ...     # Process batch of individual records
        ...     pass

    Note:
        For best performance with multi-worker DataLoader, the underlying
        Rust implementation handles file reading efficiently.
    """

    def __init__(self, file_path: str):
        """Initialize FastaDataset with file path."""
        from deepbiop.fa import FastaStreamDataset

        self.file_path = file_path

        # Use Rust FastaStreamDataset which provides efficient streaming
        self._rust_dataset = FastaStreamDataset(file_path)

        # Get total count by reading the file once
        # This is needed for __len__ and random access
        self._records_cache = list(self._read_all_records())
        self._total_records = len(self._records_cache)

    def _read_all_records(self) -> Iterator[dict[str, Any]]:
        """Read all records from file using Rust streaming dataset."""
        # Iterate through Rust dataset (already returns dicts)
        yield from self._rust_dataset

    def __len__(self) -> int:
        """Return total number of records."""
        return self._total_records

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get individual record at index.

        Args:
            idx (int): Record index (0-based)

        Returns:
            Record dict with keys: "id", "sequence", "description" (optional)
        """
        if idx < 0 or idx >= self._total_records:
            msg = f"Index {idx} out of range [0, {self._total_records})"
            raise IndexError(msg)

        return self._records_cache[idx]

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over all records.

        Yields:
            Record dict with keys: "id", "sequence", "description" (optional)
        """
        return iter(self._records_cache)

    def __repr__(self) -> str:
        """String representation."""
        return f"FastaDataset(file_path='{self.file_path}', total_records={self._total_records})"


class BamDataset:
    """Simple PyTorch-compatible BAM dataset that returns individual alignment records.

    This implementation uses a simple streaming approach with optional caching
    for better performance with PyTorch DataLoader.

    Parameters
    ----------
        file_path: Path to BAM file
        threads: Optional number of threads for bgzf decompression (None = use all available)

    Example:
        >>> from torch.utils.data import DataLoader
        >>> dataset = BamDataset("alignments.bam", threads=4)
        >>> loader = DataLoader(dataset, batch_size=32)
        >>> for batch in loader:
        ...     # Process batch of alignment records
        ...     pass

    Note:
        For best performance with multi-worker DataLoader, the underlying
        Rust implementation handles file reading efficiently with parallel decompression.
    """

    def __init__(self, file_path: str, threads: int | None = None):
        """Initialize BamDataset with file path and optional thread count."""
        from deepbiop.bam import BamStreamDataset

        self.file_path = file_path
        self.threads = threads

        # Use Rust BamStreamDataset which provides efficient streaming
        self._rust_dataset = BamStreamDataset(file_path, threads)

        # Get total count by reading the file once
        # This is needed for __len__ and random access
        self._records_cache = list(self._read_all_records())
        self._total_records = len(self._records_cache)

    def _read_all_records(self) -> Iterator[dict[str, Any]]:
        """Read all records from file using Rust streaming dataset."""
        # Iterate through Rust dataset (already returns dicts)
        yield from self._rust_dataset

    def __len__(self) -> int:
        """Return total number of records."""
        return self._total_records

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get individual record at index.

        Args:
            idx (int): Record index (0-based)

        Returns:
            Record dict with keys: "id", "sequence", "quality"
        """
        if idx < 0 or idx >= self._total_records:
            msg = f"Index {idx} out of range [0, {self._total_records})"
            raise IndexError(msg)

        return self._records_cache[idx]

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over all records.

        Yields:
            Record dict with keys: "id", "sequence", "quality"
        """
        return iter(self._records_cache)

    def __repr__(self) -> str:
        """String representation."""
        threads_str = f", threads={self.threads}" if self.threads is not None else ""
        return f"BamDataset(file_path='{self.file_path}'{threads_str}, total_records={self._total_records})"


__all__ = ["BamDataset", "FastaDataset", "FastqDataset"]
