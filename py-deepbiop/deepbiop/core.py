"""Core data structures and types for DeepBioP."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Record:
    """Unified sequence record representation.

    This is the core data structure that flows through the entire pipeline.
    All dataset implementations return Record objects, and all transforms
    operate on Record objects.

    Parameters
    ----------
    id : bytes
        Record identifier (e.g., FASTQ header, FASTA ID)
    sequence : bytes
        Biological sequence (DNA, RNA, or protein)
    quality : bytes | None
        Quality scores for the sequence (Phred-encoded). None for FASTA/BAM.
    metadata : dict[str, Any]
        Additional metadata extracted from headers or added by transforms
    features : dict[str, Any] | None
        Encoded features added by transforms (e.g., one-hot, k-mer vectors)

    Examples:
    --------
    >>> record = Record(
    ...     id=b"@read_001",
    ...     sequence=b"ACGT",
    ...     quality=b"IIII",
    ...     metadata={"sample": "test"},
    ... )
    >>> record.id
    b'@read_001'
    """

    id: bytes
    sequence: bytes
    quality: bytes | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    features: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """Validate record after initialization."""
        if not isinstance(self.id, bytes):
            msg = f"id must be bytes, got {type(self.id)}"
            raise TypeError(msg)
        if not isinstance(self.sequence, bytes):
            msg = f"sequence must be bytes, got {type(self.sequence)}"
            raise TypeError(msg)
        if self.quality is not None and not isinstance(self.quality, bytes):
            msg = f"quality must be bytes or None, got {type(self.quality)}"
            raise TypeError(msg)
        if self.quality is not None and len(self.quality) != len(self.sequence):
            msg = (
                f"quality length ({len(self.quality)}) must match "
                f"sequence length ({len(self.sequence)})"
            )
            raise ValueError(msg)

    def __len__(self) -> int:
        """Return sequence length."""
        return len(self.sequence)

    def to_dict(self) -> dict[str, Any]:
        """Convert record to dictionary representation.

        Returns:
        -------
        dict[str, Any]
            Dictionary with all record fields

        Examples:
        --------
        >>> record = Record(id=b"@r1", sequence=b"ACGT", quality=b"IIII")
        >>> record.to_dict()
        {'id': b'@r1', 'sequence': b'ACGT', 'quality': b'IIII', 'metadata': {}, 'features': None}
        """
        return {
            "id": self.id,
            "sequence": self.sequence,
            "quality": self.quality,
            "metadata": self.metadata.copy(),
            "features": self.features.copy() if self.features else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Record:
        """Create record from dictionary.

        Parameters
        ----------
        data : dict[str, Any]
            Dictionary containing record fields

        Returns:
        -------
        Record
            New record instance

        Examples:
        --------
        >>> data = {"id": b"@r1", "sequence": b"ACGT", "quality": b"IIII"}
        >>> record = Record.from_dict(data)
        >>> record.id
        b'@r1'
        """
        return cls(
            id=data["id"],
            sequence=data["sequence"],
            quality=data.get("quality"),
            metadata=data.get("metadata", {}),
            features=data.get("features"),
        )

    def copy(self) -> Record:
        """Create a deep copy of the record.

        Returns:
        -------
        Record
            New record instance with copied data
        """
        return Record(
            id=self.id,
            sequence=self.sequence,
            quality=self.quality,
            metadata=self.metadata.copy(),
            features=self.features.copy() if self.features else None,
        )
