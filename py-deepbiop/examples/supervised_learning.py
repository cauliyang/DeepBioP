"""
Supervised Learning Examples with DeepBioP.

This module demonstrates how to use DeepBioP for supervised learning tasks
with biological sequence data (FASTQ, FASTA, BAM files).

Examples include:
1. Quality score prediction (regression)
2. Sequence classification from headers
3. Multi-class classification with external labels
4. PyTorch Lightning integration
5. Custom target extraction
"""

# ===================================
# Example 1: Quality Score Prediction
# ===================================

def example_quality_prediction():
    """
    Regression task: Predict mean quality score from sequence.

    This example shows how to:
    - Use built-in quality extractors
    - Apply sequence encoding (OneHotEncoder)
    - Create DataLoader with tensor collation
    - Train a simple PyTorch model
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    from deepbiop.fq import FastqStreamDataset, OneHotEncoder
    from deepbiop.transforms import TransformDataset
    from deepbiop.targets import TargetExtractor
    from deepbiop.collate import tensor_collate

    # Create dataset with quality score as target
    base_dataset = FastqStreamDataset("data.fastq")

    dataset = TransformDataset(
        base_dataset,
        transform=OneHotEncoder(),  # Encode sequences to one-hot
        target_fn=TargetExtractor.from_quality(stat="mean"),  # Extract mean quality
        return_dict=False,  # Return (features, target) tuples
    )

    # Create DataLoader with tensor collation
    loader = DataLoader(
        dataset,
        batch_size=32,
        collate_fn=tensor_collate,
    )

    # Simple regression model
    class QualityPredictor(nn.Module):
        def __init__(self, seq_length=100):
            super().__init__()
            self.conv1 = nn.Conv1d(4, 64, kernel_size=7)  # 4 channels for ACGT
            self.conv2 = nn.Conv1d(64, 128, kernel_size=5)
            self.pool = nn.AdaptiveMaxPool1d(1)
            self.fc = nn.Linear(128, 1)

        def forward(self, x):
            # x shape: (batch, seq_length, 4) -> transpose to (batch, 4, seq_length)
            x = x.transpose(1, 2)
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = self.pool(x).squeeze(-1)
            return self.fc(x).squeeze(-1)

    # Training loop
    model = QualityPredictor()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(5):
        total_loss = 0.0
        for batch in loader:
            features = batch["features"]  # Shape: (batch_size, seq_length, 4)
            targets = batch["targets"]    # Shape: (batch_size,)

            # Forward pass
            predictions = model(features)
            loss = criterion(predictions, targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")


# ===================================
# Example 2: Sequence Classification from Headers
# ===================================

def example_header_classification():
    """
    Classification task: Extract labels from FASTQ headers.

    Headers format: @read_123|class:positive|score:0.95

    This example shows how to:
    - Parse structured headers
    - Create multi-class classification dataset
    - Use categorical cross-entropy loss
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    from deepbiop.fq import FastqStreamDataset
    from deepbiop.transforms import TransformDataset
    from deepbiop.targets import create_classification_extractor
    from deepbiop import KmerEncoder  # K-mer encoding
    from deepbiop.collate import tensor_collate

    # Define classes
    classes = ["negative", "positive"]

    # Create classifier extractor
    target_extractor = create_classification_extractor(
        classes=classes,
        key="class",  # Extract from key:value pairs
    )

    # Create dataset
    base_dataset = FastqStreamDataset("labeled_reads.fastq")

    dataset = TransformDataset(
        base_dataset,
        transform=KmerEncoder(k=6),  # 6-mer encoding
        target_fn=target_extractor,
        return_dict=False,
    )

    # DataLoader
    loader = DataLoader(dataset, batch_size=64, collate_fn=tensor_collate)

    # Classification model
    class SequenceClassifier(nn.Module):
        def __init__(self, input_dim, num_classes=2):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, 256)
            self.fc2 = nn.Linear(256, 128)
            self.fc3 = nn.Linear(128, num_classes)
            self.dropout = nn.Dropout(0.3)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.dropout(x)
            x = torch.relu(self.fc2(x))
            x = self.dropout(x)
            return self.fc3(x)

    # Training
    model = SequenceClassifier(input_dim=4096, num_classes=len(classes))  # 4^6 = 4096 for 6-mers
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(10):
        for batch in loader:
            features = batch["features"]
            targets = batch["targets"].long()

            predictions = model(features)
            loss = criterion(predictions, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


# ===================================
# Example 3: External Label File
# ===================================

def example_external_labels():
    """
    Classification with external CSV label file.

    labels.csv:
    read_id,class,confidence
    read_1,0,0.95
    read_2,1,0.88
    read_3,0,0.92

    This example shows how to:
    - Load labels from external file
    - Map sequence IDs to labels
    - Handle missing labels gracefully
    """
    from deepbiop.fq import FastqStreamDataset, OneHotEncoder
    from deepbiop.transforms import TransformDataset
    from deepbiop.targets import TargetExtractor

    # Create target extractor from CSV file
    target_extractor = TargetExtractor.from_file(
        filepath="labels.csv",
        id_column="read_id",
        label_column="class",
        converter=int,  # Convert label to integer
    )

    # Create dataset
    base_dataset = FastqStreamDataset("reads.fastq")

    dataset = TransformDataset(
        base_dataset,
        transform=OneHotEncoder(),
        target_fn=target_extractor,
        return_dict=True,  # Keep as dict for inspection
    )

    # Iterate
    for sample in dataset:
        seq_id = sample["id"]
        features = sample["features"]  # Encoded sequence
        target = sample["target"]      # Label from CSV
        print(f"ID: {seq_id}, Target: {target}, Features shape: {features.shape}")
        break  # Just show first sample


# ===================================
# Example 4: PyTorch Lightning Integration
# ===================================

def example_lightning_training():
    """
    Full training pipeline with PyTorch Lightning.

    This example shows how to:
    - Use BiologicalDataModule
    - Create LightningModule
    - Train with Trainer
    - Validate and test
    """
    try:
        import pytorch_lightning as pl
    except ImportError:
        print("PyTorch Lightning not installed. Skipping this example.")
        return

    import torch
    import torch.nn as nn
    from deepbiop.lightning import BiologicalDataModule
    from deepbiop.targets import TargetExtractor
    from deepbiop import OneHotEncoder

    # Data Module
    data_module = BiologicalDataModule(
        train_path="train.fastq",
        val_path="val.fastq",
        test_path="test.fastq",
        transform=OneHotEncoder(),
        target_fn=TargetExtractor.from_quality("mean"),
        collate_mode="tensor",
        batch_size=32,
        num_workers=4,
    )

    # Lightning Module
    class QualityPredictor(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv1d(4, 64, kernel_size=7)
            self.conv2 = nn.Conv1d(64, 128, kernel_size=5)
            self.pool = nn.AdaptiveMaxPool1d(1)
            self.fc = nn.Linear(128, 1)
            self.criterion = nn.MSELoss()

        def forward(self, x):
            x = x.transpose(1, 2)  # (batch, seq, 4) -> (batch, 4, seq)
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = self.pool(x).squeeze(-1)
            return self.fc(x).squeeze(-1)

        def training_step(self, batch, batch_idx):
            features = batch["features"]
            targets = batch["targets"]
            predictions = self(features)
            loss = self.criterion(predictions, targets)
            self.log("train_loss", loss)
            return loss

        def validation_step(self, batch, batch_idx):
            features = batch["features"]
            targets = batch["targets"]
            predictions = self(features)
            loss = self.criterion(predictions, targets)
            self.log("val_loss", loss)
            return loss

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=0.001)

    # Train
    model = QualityPredictor()
    trainer = pl.Trainer(
        max_epochs=10,
        accelerator="auto",
        devices=1,
    )

    trainer.fit(model, data_module)
    trainer.test(model, data_module)


# ===================================
# Example 5: Custom Target Extraction
# ===================================

def example_custom_extractor():
    """
    Custom target extraction function.

    This example shows how to:
    - Write custom extraction logic
    - Combine multiple features as targets
    - Handle complex label formats
    """
    from deepbiop.fq import FastqStreamDataset
    from deepbiop.transforms import TransformDataset
    from deepbiop.targets import TargetExtractor
    from deepbiop import OneHotEncoder

    # Custom extraction function
    def extract_gc_and_quality(record):
        """
        Extract both GC content and quality score as multi-target.

        Returns tuple of (gc_content, mean_quality)
        """
        # GC content
        seq = record["sequence"]
        gc = seq.count(b'G') + seq.count(b'C')
        gc_content = gc / len(seq) if len(seq) > 0 else 0.0

        # Mean quality
        quality = record.get("quality", [])
        mean_quality = sum(quality) / len(quality) if quality else 0.0

        return (gc_content, mean_quality)

    # Create extractor
    target_extractor = TargetExtractor(extract_gc_and_quality)

    # Create dataset
    base_dataset = FastqStreamDataset("data.fastq")

    dataset = TransformDataset(
        base_dataset,
        transform=OneHotEncoder(),
        target_fn=target_extractor,
        return_dict=True,
    )

    # Iterate
    for sample in dataset:
        gc, quality = sample["target"]
        print(f"GC: {gc:.3f}, Quality: {quality:.1f}")
        break


# ===================================
# Example 6: Multi-Task Learning
# ===================================

def example_multitask_learning():
    """
    Multi-task learning with multiple targets.

    Predict both quality score (regression) and read type (classification).
    """
    import torch
    import torch.nn as nn
    from deepbiop.fq import FastqStreamDataset
    from deepbiop.transforms import TransformDataset
    from deepbiop.targets import TargetExtractor
    from deepbiop import OneHotEncoder

    # Multi-target extractor
    def extract_multi_targets(record):
        """Return dict with multiple targets."""
        # Quality score (regression)
        quality = record.get("quality", [])
        mean_quality = sum(quality) / len(quality) if quality else 0.0

        # Read type from header (classification)
        header = record["id"].decode() if isinstance(record["id"], bytes) else record["id"]
        read_type = 1 if "type:good" in header else 0

        return {
            "quality": mean_quality,
            "type": read_type,
        }

    dataset = TransformDataset(
        FastqStreamDataset("data.fastq"),
        transform=OneHotEncoder(),
        target_fn=TargetExtractor(extract_multi_targets),
        return_dict=True,
    )

    # Multi-task model
    class MultiTaskModel(nn.Module):
        def __init__(self):
            super().__init__()
            # Shared encoder
            self.encoder = nn.Sequential(
                nn.Conv1d(4, 64, kernel_size=7),
                nn.ReLU(),
                nn.Conv1d(64, 128, kernel_size=5),
                nn.ReLU(),
                nn.AdaptiveMaxPool1d(1),
            )
            # Task-specific heads
            self.quality_head = nn.Linear(128, 1)  # Regression
            self.type_head = nn.Linear(128, 2)      # Binary classification

        def forward(self, x):
            x = x.transpose(1, 2)
            features = self.encoder(x).squeeze(-1)

            quality_pred = self.quality_head(features).squeeze(-1)
            type_pred = self.type_head(features)

            return {
                "quality": quality_pred,
                "type": type_pred,
            }

    # Training would involve combining losses
    model = MultiTaskModel()
    quality_loss = nn.MSELoss()
    type_loss = nn.CrossEntropyLoss()

    # Example forward pass
    sample = next(iter(dataset))
    features = torch.tensor(sample["features"]).unsqueeze(0)
    predictions = model(features)
    print(f"Quality pred: {predictions['quality'].item():.2f}")
    print(f"Type logits: {predictions['type']}")


if __name__ == "__main__":
    print("DeepBioP Supervised Learning Examples")
    print("=" * 50)

    print("\n1. Quality Score Prediction (Regression)")
    print("-" * 50)
    # Uncomment to run:
    # example_quality_prediction()

    print("\n2. Header-based Classification")
    print("-" * 50)
    # example_header_classification()

    print("\n3. External Label File")
    print("-" * 50)
    # example_external_labels()

    print("\n4. PyTorch Lightning Integration")
    print("-" * 50)
    # example_lightning_training()

    print("\n5. Custom Target Extraction")
    print("-" * 50)
    # example_custom_extractor()

    print("\n6. Multi-Task Learning")
    print("-" * 50)
    # example_multitask_learning()

    print("\nAll examples defined. Uncomment to run specific examples.")
