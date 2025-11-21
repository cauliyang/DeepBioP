"""PyTorch Lightning integration for biological sequence data.

Provides LightningDataModule implementation for seamless integration with
PyTorch Lightning training workflows.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

from pytorch_lightning import LightningDataModule as _LightningDataModule
from torch.utils.data import DataLoader

__all__ = ["BiologicalDataModule"]

class BiologicalDataModule(_LightningDataModule):
    """PyTorch Lightning DataModule for biological sequence data.

    Handles data loading, preprocessing, and splitting for train/val/test sets
    with PyTorch Lightning integration.

    Args:
        train_files: Path(s) to training data file(s)
        val_files: Optional path(s) to validation data file(s)
        test_files: Optional path(s) to test data file(s)
        batch_size: Batch size for data loaders (default: 32)
        num_workers: Number of worker processes for data loading (default: 0)
        shuffle_train: Whether to shuffle training data (default: True)
        transform: Optional transformation pipeline to apply
        collate_fn: Optional custom collate function

    Example:
        >>> from deepbiop.lightning import BiologicalDataModule
        >>> from deepbiop.pytorch import OneHotEncoder
        >>> dm = BiologicalDataModule(
        ...     train_files="train.fastq",
        ...     val_files="val.fastq",
        ...     batch_size=32,
        ...     transform=OneHotEncoder(encoding_type="dna"),
        ... )
        >>> # Use with PyTorch Lightning Trainer
        >>> trainer.fit(model, dm)
    """

    def __init__(
        self,
        train_files: str | list[str] | Path | list[Path],
        val_files: str | list[str] | Path | list[Path] | None = None,
        test_files: str | list[str] | Path | list[Path] | None = None,
        *,
        batch_size: int = 32,
        num_workers: int = 0,
        shuffle_train: bool = True,
        transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        collate_fn: Callable[[list[dict[str, Any]]], Any] | None = None,
    ) -> None: ...
    def prepare_data(self) -> None:
        """Download and prepare data (called once per node)."""
    def setup(self, stage: str | None = None) -> None:
        """Set up datasets for train/val/test splits.

        Args:
            stage: 'fit', 'test', or None (for both)
        """
    def train_dataloader(self) -> DataLoader:
        """Return training DataLoader."""
    def val_dataloader(self) -> DataLoader:
        """Return validation DataLoader."""
    def test_dataloader(self) -> DataLoader:
        """Return test DataLoader."""
