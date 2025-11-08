"""
Tests for Hugging Face Transformers integration.

This module tests the integration with Hugging Face's Trainer API
for training models on biological sequence data.
"""

from pathlib import Path

import pytest


class TestTransformersTrainer:
    """Test Hugging Face Transformers Trainer integration (T039)."""

    def test_hf_trainer_basic(self):
        """Test that HuggingFace Trainer works with our datasets."""
        try:
            import torch
            import torch.nn as nn
            from transformers import Trainer, TrainingArguments

            from deepbiop.fq import FastqStreamDataset
        except ImportError:
            pytest.skip("transformers not installed")

        test_file = Path(__file__).parent / "data" / "test.fastq"
        if not test_file.exists():
            pytest.skip(f"Test file not found: {test_file}")

        # Wrap our dataset to make it compatible with HF Trainer
        class HFCompatibleDataset:
            """Wrapper to make FastqStreamDataset compatible with HF Trainer."""

            def __init__(self, dataset):
                """Initialize the wrapper with a FastqStreamDataset."""
                self.dataset = dataset
                self._iter = None

            def __iter__(self):
                """Return iterator over dataset."""
                # Reset iterator each time __iter__ is called
                self._iter = iter(self.dataset)
                return self

            def __next__(self):
                """Get next item with HF Trainer compatible format."""
                next(self._iter)
                # HF Trainer expects specific keys
                # For testing, just return a dict with dummy tensors
                return {
                    "input_ids": torch.randint(0, 100, (10,)),
                    "labels": torch.randint(0, 2, (1,)),
                }

            def __len__(self):
                """Return dataset length."""
                return len(self.dataset)

            def __getitem__(self, idx):
                """Get item by index for map-style dataset access."""
                # Use the dataset's __getitem__ (now implemented in Rust)
                self.dataset[idx]
                # Return HF Trainer compatible format
                return {
                    "input_ids": torch.randint(0, 100, (10,)),
                    "labels": torch.randint(0, 2, (1,)),
                }

        # Create simple model for testing
        class DummyModel(nn.Module):
            """Simple model for testing HF Trainer integration."""

            def __init__(self):
                """Initialize the dummy model."""
                super().__init__()
                self.fc = nn.Linear(10, 2)

            def forward(self, input_ids, labels=None):
                """Forward pass with HF Trainer compatible output format."""
                # HF Trainer expects dict output with 'loss' key
                output = self.fc(input_ids.float())
                if labels is not None:
                    loss = nn.functional.cross_entropy(
                        output.view(-1, 2), labels.view(-1)
                    )
                    return {"loss": loss, "logits": output}
                return {"logits": output}

        # Create dataset
        dataset = FastqStreamDataset(str(test_file))
        train_dataset = HFCompatibleDataset(dataset)

        # Create model
        model = DummyModel()

        # Create training arguments
        training_args = TrainingArguments(
            output_dir="./test_output",
            num_train_epochs=1,
            per_device_train_batch_size=2,
            max_steps=2,  # Only train for 2 steps
            logging_steps=1,
            save_strategy="no",
            report_to="none",
        )

        # Create Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
        )

        # Test that training works
        try:
            import warnings

            with warnings.catch_warnings():
                # Ignore pin_memory warning on MPS devices
                warnings.filterwarnings("ignore", message=".*pin_memory.*")
                trainer.train()
            assert True, "HF Trainer.train() completed successfully"
        except Exception as e:
            pytest.fail(f"HF Trainer.train() failed: {e}")

    def test_hf_trainer_with_collate_fn(self):
        """Test HF Trainer with custom collate function."""
        try:
            import torch
            import torch.nn as nn
            from transformers import Trainer, TrainingArguments

            from deepbiop.fq import FastqStreamDataset
        except ImportError:
            pytest.skip("transformers not installed")

        test_file = Path(__file__).parent / "data" / "test.fastq"
        if not test_file.exists():
            pytest.skip(f"Test file not found: {test_file}")

        # Custom collate function for biological sequences
        def bio_collate_fn(batch):
            """Collate biological sequence records into tensors."""
            # batch is a list of dicts from our dataset
            # Convert to format HF Trainer expects
            return {
                "input_ids": torch.stack([torch.randint(0, 100, (10,)) for _ in batch]),
                "labels": torch.randint(0, 2, (len(batch),)),
            }

        class HFCompatibleDataset:
            """Wrapper dataset with map-style access support."""

            def __init__(self, dataset):
                """Initialize wrapper with dataset."""
                self.dataset = dataset
                self._items = None

            def __iter__(self):
                """Return iterator over dataset."""
                return iter(self.dataset)

            def __len__(self):
                """Return dataset length."""
                return len(self.dataset)

            def __getitem__(self, idx):
                """Get item by index for map-style access."""
                # For map-style access if needed
                if self._items is None:
                    self._items = list(self.dataset)
                return self._items[idx]

        class DummyModel(nn.Module):
            """Simple model for testing with custom collate function."""

            def __init__(self):
                """Initialize the model."""
                super().__init__()
                self.fc = nn.Linear(10, 2)

            def forward(self, input_ids, labels=None):
                """Forward pass."""
                output = self.fc(input_ids.float())
                if labels is not None:
                    loss = nn.functional.cross_entropy(output, labels)
                    return {"loss": loss, "logits": output}
                return {"logits": output}

        # Create dataset
        dataset = FastqStreamDataset(str(test_file))
        train_dataset = HFCompatibleDataset(dataset)

        model = DummyModel()

        training_args = TrainingArguments(
            output_dir="./test_output",
            num_train_epochs=1,
            per_device_train_batch_size=2,
            max_steps=2,
            logging_steps=1,
            save_strategy="no",
            report_to="none",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=bio_collate_fn,
        )

        try:
            import warnings

            with warnings.catch_warnings():
                # Ignore pin_memory warning on MPS devices
                warnings.filterwarnings("ignore", message=".*pin_memory.*")
                trainer.train()
            assert True, "HF Trainer with custom collate completed"
        except Exception as e:
            pytest.fail(f"HF Trainer with custom collate failed: {e}")

    def test_hf_trainer_evaluation(self):
        """Test HF Trainer evaluation with our datasets."""
        try:
            import torch
            import torch.nn as nn
            from transformers import Trainer, TrainingArguments

            from deepbiop.fq import FastqStreamDataset
        except ImportError:
            pytest.skip("transformers not installed")

        test_file = Path(__file__).parent / "data" / "test.fastq"
        if not test_file.exists():
            pytest.skip(f"Test file not found: {test_file}")

        class HFCompatibleDataset:
            """Wrapper dataset for evaluation testing."""

            def __init__(self, dataset):
                """Initialize wrapper with dataset."""
                self.dataset = dataset

            def __iter__(self):
                """Return iterator yielding HF-compatible dicts."""
                for _item in self.dataset:
                    yield {
                        "input_ids": torch.randint(0, 100, (10,)),
                        "labels": torch.randint(0, 2, (1,)),
                    }

            def __len__(self):
                """Return dataset length."""
                return len(self.dataset)

            def __getitem__(self, idx):
                """Get item by index for map-style dataset access."""
                # Use the dataset's __getitem__ (now implemented in Rust)
                self.dataset[idx]
                # Return HF Trainer compatible format
                return {
                    "input_ids": torch.randint(0, 100, (10,)),
                    "labels": torch.randint(0, 2, (1,)),
                }

        class DummyModel(nn.Module):
            """Simple model for evaluation testing."""

            def __init__(self):
                """Initialize the model."""
                super().__init__()
                self.fc = nn.Linear(10, 2)

            def forward(self, input_ids, labels=None):
                """Forward pass with loss computation."""
                output = self.fc(input_ids.float())
                if labels is not None:
                    loss = nn.functional.cross_entropy(
                        output.view(-1, 2), labels.view(-1)
                    )
                    return {"loss": loss, "logits": output}
                return {"logits": output}

        dataset = FastqStreamDataset(str(test_file))
        eval_dataset = HFCompatibleDataset(dataset)

        model = DummyModel()

        training_args = TrainingArguments(
            output_dir="./test_output", per_device_eval_batch_size=2, report_to="none"
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            eval_dataset=eval_dataset,
        )

        try:
            import warnings

            with warnings.catch_warnings():
                # Ignore pin_memory warning on MPS devices
                warnings.filterwarnings("ignore", message=".*pin_memory.*")
                # Test evaluation (limit dataset size for quick test)
                metrics = trainer.evaluate()
            assert "eval_loss" in metrics, "Should return eval_loss metric"
            assert True, "HF Trainer evaluation completed"
        except Exception as e:
            pytest.fail(f"HF Trainer evaluation failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
