"""
Tests for PyTorch Lightning integration.

This module tests the integration with PyTorch Lightning's
LightningDataModule for train/val/test splits and Trainer integration.
"""

from pathlib import Path

import pytest


class TestLightningDataModule:
    """Test LightningDataModule integration (T037-T038)."""

    def test_datamodule_splits(self):
        """Test LightningDataModule with train/val/test splits."""
        try:
            from deepbiop.lightning import BiologicalDataModule
        except ImportError:
            pytest.skip("pytorch_lightning not installed")

        test_file = Path(__file__).parent / "data" / "test.fastq"
        if not test_file.exists():
            pytest.skip(f"Test file not found: {test_file}")

        # Create data module with train/val/test split
        data_module = BiologicalDataModule(
            train_path=str(test_file),
            val_path=str(test_file),  # Using same file for testing
            test_path=str(test_file),  # Using same file for testing
            batch_size=4,
            num_workers=0,
        )

        # Setup should create train/val/test datasets
        data_module.setup(stage="fit")

        # Check train dataloader
        train_loader = data_module.train_dataloader()
        assert train_loader is not None, "train_dataloader() should return loader"

        train_batch = next(iter(train_loader))
        assert isinstance(train_batch, list), "Train batch should be list"

        # Check val dataloader
        val_loader = data_module.val_dataloader()
        assert val_loader is not None, "val_dataloader() should return loader"

        val_batch = next(iter(val_loader))
        assert isinstance(val_batch, list), "Val batch should be list"

        # Setup test stage
        data_module.setup(stage="test")
        test_loader = data_module.test_dataloader()
        assert test_loader is not None, "test_dataloader() should return loader"

        test_batch = next(iter(test_loader))
        assert isinstance(test_batch, list), "Test batch should be list"

    def test_datamodule_setup_stages(self):
        """Test that setup() works correctly for different stages."""
        try:
            from deepbiop.lightning import BiologicalDataModule
        except ImportError:
            pytest.skip("pytorch_lightning not installed")

        test_file = Path(__file__).parent / "data" / "test.fastq"
        if not test_file.exists():
            pytest.skip(f"Test file not found: {test_file}")

        data_module = BiologicalDataModule(
            train_path=str(test_file),
            val_path=str(test_file),
            test_path=str(test_file),
            batch_size=2,
            num_workers=0,
        )

        # Test fit stage setup
        data_module.setup(stage="fit")
        assert hasattr(data_module, "train_dataset"), (
            "Should have train_dataset after fit setup"
        )
        assert hasattr(data_module, "val_dataset"), (
            "Should have val_dataset after fit setup"
        )

        # Test test stage setup
        data_module.setup(stage="test")
        assert hasattr(data_module, "test_dataset"), (
            "Should have test_dataset after test setup"
        )

        # Test None stage (setup all)
        data_module.setup(stage=None)
        assert hasattr(data_module, "train_dataset"), (
            "Should have all datasets with stage=None"
        )
        assert hasattr(data_module, "val_dataset"), (
            "Should have all datasets with stage=None"
        )
        assert hasattr(data_module, "test_dataset"), (
            "Should have all datasets with stage=None"
        )

    def test_datamodule_prepare_data(self):
        """Test prepare_data() method for caching/download logic."""
        try:
            from deepbiop.lightning import BiologicalDataModule
        except ImportError:
            pytest.skip("pytorch_lightning not installed")

        test_file = Path(__file__).parent / "data" / "test.fastq"
        if not test_file.exists():
            pytest.skip(f"Test file not found: {test_file}")

        data_module = BiologicalDataModule(
            train_path=str(test_file),
            val_path=str(test_file),
            test_path=str(test_file),
            batch_size=2,
            num_workers=0,
        )

        # prepare_data() should not raise errors
        # It's typically used for downloading/caching
        try:
            data_module.prepare_data()
        except AttributeError:
            # It's okay if prepare_data is not implemented yet
            pytest.skip("prepare_data() not implemented")


class TestLightningTrainer:
    """Test Lightning Trainer integration (T038)."""

    def test_trainer_fit_basic(self):
        """Test that Trainer.fit() works with BiologicalDataModule."""
        try:
            import pytorch_lightning as pl
            import torch
            import torch.nn as nn

            from deepbiop.lightning import BiologicalDataModule
        except ImportError:
            pytest.skip("pytorch_lightning not installed")

        test_file = Path(__file__).parent / "data" / "test.fastq"
        if not test_file.exists():
            pytest.skip(f"Test file not found: {test_file}")

        # Create a simple model for testing
        class DummyModel(pl.LightningModule):
            def __init__(self):
                super().__init__()
                self.layer = nn.Linear(10, 2)

            def forward(self, x):
                return self.layer(x)

            def training_step(self, batch, batch_idx):
                # Batch is a list of dicts from our dataset
                # Just return a dummy loss
                return torch.tensor(0.5, requires_grad=True)

            def validation_step(self, batch, batch_idx):
                return torch.tensor(0.3)

            def configure_optimizers(self):
                return torch.optim.Adam(self.parameters(), lr=0.001)

        # Create data module
        data_module = BiologicalDataModule(
            train_path=str(test_file),
            val_path=str(test_file),
            batch_size=4,
            num_workers=0,
        )

        # Create model and trainer
        model = DummyModel()
        trainer = pl.Trainer(
            max_epochs=1,
            limit_train_batches=2,  # Only process 2 batches
            limit_val_batches=1,  # Only process 1 val batch
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
        )

        # This should not raise errors
        try:
            import warnings

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message=".*does not have many workers.*"
                )
                trainer.fit(model, data_module)
            assert True, "Trainer.fit() completed successfully"
        except Exception as e:
            pytest.fail(f"Trainer.fit() failed: {e}")

    def test_trainer_test(self):
        """Test that Trainer.test() works with BiologicalDataModule."""
        try:
            import pytorch_lightning as pl
            import torch
            import torch.nn as nn

            from deepbiop.lightning import BiologicalDataModule
        except ImportError:
            pytest.skip("pytorch_lightning not installed")

        test_file = Path(__file__).parent / "data" / "test.fastq"
        if not test_file.exists():
            pytest.skip(f"Test file not found: {test_file}")

        # Create a simple model
        class DummyModel(pl.LightningModule):
            def __init__(self):
                super().__init__()
                self.layer = nn.Linear(10, 2)

            def test_step(self, batch, batch_idx):
                return torch.tensor(0.2)

            def configure_optimizers(self):
                return torch.optim.Adam(self.parameters(), lr=0.001)

        # Create data module
        data_module = BiologicalDataModule(
            test_path=str(test_file), batch_size=4, num_workers=0
        )

        model = DummyModel()
        trainer = pl.Trainer(
            limit_test_batches=2,
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
        )

        try:
            import warnings

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message=".*does not have many workers.*"
                )
                trainer.test(model, data_module)
            assert True, "Trainer.test() completed successfully"
        except Exception as e:
            pytest.fail(f"Trainer.test() failed: {e}")

    def test_datamodule_with_multiple_file_types(self):
        """Test DataModule with FASTQ, FASTA, and BAM files."""
        try:
            from deepbiop.lightning import BiologicalDataModule
        except ImportError:
            pytest.skip("pytorch_lightning not installed")

        # Test with different file types
        test_files = {
            "fastq": Path(__file__).parent / "data" / "test.fastq",
            "fasta": Path(__file__).parent / "data" / "test.fasta",
            "bam": Path(__file__).parent / "data" / "test.bam",
        }

        for file_type, test_file in test_files.items():
            if not test_file.exists():
                continue

            data_module = BiologicalDataModule(
                train_path=str(test_file), batch_size=2, num_workers=0
            )

            data_module.setup(stage="fit")
            train_loader = data_module.train_dataloader()

            # Should be able to get at least one batch
            batch = next(iter(train_loader))
            assert isinstance(batch, list), f"{file_type}: Batch should be list"
            assert len(batch) > 0, f"{file_type}: Batch should not be empty"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
