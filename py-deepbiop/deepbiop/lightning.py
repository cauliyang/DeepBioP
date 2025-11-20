"""
PyTorch Lightning integration for biological data.

This module provides LightningDataModule for seamless integration
with PyTorch Lightning training workflows, supporting both unsupervised
and supervised learning with easy target extraction.
"""

from pathlib import Path
from typing import Any, Callable


def _identity_collate(batch):
    """
    Identity collate function for variable-length biological sequences.

    This function can be pickled for multiprocessing (unlike lambda).

    Args:
        batch: List of samples from the dataset

    Returns
    -------
        The batch as-is (list of samples)
    """
    return batch


try:
    import pytorch_lightning as pl

    HAS_LIGHTNING = True
except ImportError:
    # Create a dummy base class if Lightning is not installed
    class pl:  # type: ignore
        """Dummy PyTorch Lightning module for when it's not installed."""

        class LightningDataModule:
            """Dummy LightningDataModule for when Lightning is not installed."""

    HAS_LIGHTNING = False


class BiologicalDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for biological sequence data.

    This class provides a Lightning-compatible interface for loading
    FASTQ, FASTA, and BAM files with automatic train/val/test splitting
    and support for supervised learning with target extraction.

    Parameters
    ----------
        train_path: Path to training data file (FASTQ, FASTA, or BAM)
        val_path: Optional path to validation data file
        test_path: Optional path to test data file
        batch_size: Batch size for DataLoaders (default: 32)
        num_workers: Number of worker processes for DataLoader (default: 0)
        file_type: File type ('fastq', 'fasta', 'bam'), auto-detected if None
        transform: Transform to apply to sequences (e.g., OneHotEncoder, KmerEncoder)
        target_fn: Function or TargetExtractor to extract targets for supervised learning
        label_file: Path to CSV/JSON file with external labels (alternative to target_fn)
        collate_mode: Collate mode ("default", "supervised", "tensor")
        return_dict: If True, return dict samples; if False, return (features, target) tuples

    Example:
        >>> # Unsupervised learning (existing behavior)
        >>> data_module = BiologicalDataModule(
        ...     train_path="train.fastq",
        ...     val_path="val.fastq",
        ...     batch_size=64,
        ... )
        >>>
        >>> # Supervised learning with quality prediction
        >>> from deepbiop.targets import TargetExtractor
        >>> data_module = BiologicalDataModule(
        ...     train_path="train.fastq",
        ...     val_path="val.fastq",
        ...     transform=OneHotEncoder(),
        ...     target_fn=TargetExtractor.from_quality("mean"),
        ...     collate_mode="tensor",
        ...     batch_size=32,
        ... )
        >>>
        >>> # Supervised learning with external labels
        >>> data_module = BiologicalDataModule(
        ...     train_path="train.fastq",
        ...     label_file="labels.csv",
        ...     transform=KmerEncoder(k=6),
        ...     batch_size=32,
        ... )
    """

    def __init__(
        self,
        train_path: str | None = None,
        val_path: str | None = None,
        test_path: str | None = None,
        batch_size: int = 32,
        num_workers: int = 0,
        file_type: str | None = None,
        transform: Any | None = None,
        target_fn: Callable | Any | None = None,
        label_file: str | None = None,
        collate_mode: str = "default",
        return_dict: bool = True,
    ):
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.file_type = file_type
        self.transform = transform
        self.target_fn = target_fn
        self.label_file = label_file
        self.collate_mode = collate_mode
        self.return_dict = return_dict

        # Create target extractor from label file if provided
        if label_file is not None and target_fn is None:
            from deepbiop.targets import TargetExtractor
            self.target_fn = TargetExtractor.from_file(label_file)

        # Datasets will be created in setup()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        """
        Called only once to download/prepare data.

        This is called on a single GPU in distributed training.
        Use this to download or prepare data that should not be
        done on every process.
        """
        # For now, we assume data is already available
        # This could be extended to download data, create caches, etc.

    def setup(self, stage: str | None = None):
        """
        Called on every process in distributed training.

        Creates train/val/test datasets based on stage.

        Args:
            stage: 'fit', 'test', 'predict', or None for all
        """
        # Auto-detect file type from extension if not specified
        file_type = self.file_type

        if stage == "fit" or stage is None:
            # Setup train dataset
            if self.train_path is not None:
                self.train_dataset = self._create_dataset(self.train_path, file_type)

            # Setup val dataset
            if self.val_path is not None:
                self.val_dataset = self._create_dataset(self.val_path, file_type)

        if stage == "test" or stage is None:
            # Setup test dataset
            if self.test_path is not None:
                self.test_dataset = self._create_dataset(self.test_path, file_type)

    def _create_dataset(self, file_path: str, file_type: str | None = None):
        """
        Create appropriate dataset based on file type with transforms and targets.

        Args:
            file_path: Path to data file
            file_type: File type ('fastq', 'fasta', 'bam'), auto-detected if None

        Returns
        -------
            Dataset wrapped with transforms and target extraction if configured
        """
        from deepbiop.transforms import TransformDataset

        path = Path(file_path)

        # Auto-detect file type from extension
        if file_type is None:
            suffix = path.suffix.lower()
            if suffix in [".fastq", ".fq"] or ".fastq" in path.name.lower():
                file_type = "fastq"
            elif suffix in [".fasta", ".fa"] or ".fasta" in path.name.lower():
                file_type = "fasta"
            elif suffix == ".bam":
                file_type = "bam"
            else:
                msg = (
                    f"Cannot auto-detect file type from {file_path}. "
                    f"Please specify file_type parameter."
                )
                raise ValueError(msg)

        # Create base dataset
        if file_type == "fastq":
            from deepbiop.fq import FastqStreamDataset

            base_dataset = FastqStreamDataset(str(file_path))
        elif file_type == "fasta":
            from deepbiop.fa import FastaStreamDataset

            base_dataset = FastaStreamDataset(str(file_path))
        elif file_type == "bam":
            from deepbiop.bam import BamStreamDataset

            base_dataset = BamStreamDataset(str(file_path))
        else:
            msg = f"Unsupported file type: {file_type}"
            raise ValueError(msg)

        # Wrap with transform and target extraction if configured
        if self.transform is not None or self.target_fn is not None:
            return TransformDataset(
                base_dataset,
                transform=self.transform,
                target_fn=self.target_fn,
                return_dict=self.return_dict,
            )

        return base_dataset

    def train_dataloader(self):
        """
        Create training DataLoader.

        Returns
        -------
            PyTorch DataLoader for training data
        """
        from torch.utils.data import DataLoader
        from deepbiop.collate import get_collate_fn

        if self.train_dataset is None:
            msg = "Train dataset not set up. Call setup(stage='fit') first."
            raise RuntimeError(msg)

        collate_fn = get_collate_fn(self.collate_mode)

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,  # Streaming datasets don't support shuffling easily
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        """
        Create validation DataLoader.

        Returns
        -------
            PyTorch DataLoader for validation data
        """
        from torch.utils.data import DataLoader
        from deepbiop.collate import get_collate_fn

        if self.val_dataset is None:
            msg = "Val dataset not set up. Call setup(stage='fit') first."
            raise RuntimeError(msg)

        collate_fn = get_collate_fn(self.collate_mode)

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=collate_fn,
        )

    def test_dataloader(self):
        """
        Create test DataLoader.

        Returns
        -------
            PyTorch DataLoader for test data
        """
        from torch.utils.data import DataLoader
        from deepbiop.collate import get_collate_fn

        if self.test_dataset is None:
            msg = "Test dataset not set up. Call setup(stage='test') first."
            raise RuntimeError(msg)

        collate_fn = get_collate_fn(self.collate_mode)

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=collate_fn,
        )
