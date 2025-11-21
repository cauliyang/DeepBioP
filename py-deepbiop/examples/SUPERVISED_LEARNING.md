# Supervised Learning with DeepBioP

DeepBioP provides a user-friendly API for supervised learning with biological sequence data, compatible with PyTorch and PyTorch Lightning.

## Quick Start

```python
from deepbiop.lightning import BiologicalDataModule
from deepbiop.targets import TargetExtractor
from deepbiop import OneHotEncoder
import pytorch_lightning as pl

# Create data module with built-in target extraction
data_module = BiologicalDataModule(
    train_path="train.fastq",
    val_path="val.fastq",
    transform=OneHotEncoder(),
    target_fn=TargetExtractor.from_quality("mean"),
    collate_mode="tensor",
    batch_size=32,
)

# Train with PyTorch Lightning
trainer = pl.Trainer(max_epochs=10)
model = YourLightningModel()
trainer.fit(model, data_module)
```

## Target Extraction Methods

### 1. From Quality Scores (FASTQ)

Extract statistics from Phred quality scores:

```python
from deepbiop.targets import TargetExtractor

# Mean quality score
extractor = TargetExtractor.from_quality(stat="mean")

# Other statistics: median, min, max, std
extractor = TargetExtractor.from_quality(stat="median")
```

### 2. From Header Metadata

#### Using Regex Patterns

```python
# Headers like: @read_123 label=positive score=0.95
extractor = TargetExtractor.from_header(
    pattern=r"label=(\w+)",
    converter=str
)
```

#### Using Key-Value Pairs

```python
# Headers like: @read_123|class:1|score:0.95
extractor = TargetExtractor.from_header(
    key="class",
    separator="|",
    converter=int
)
```

### 3. From Sequence Features

```python
# GC content
extractor = TargetExtractor.from_sequence(feature="gc_content")

# Sequence length
extractor = TargetExtractor.from_sequence(feature="length")

# Sequence complexity
extractor = TargetExtractor.from_sequence(feature="complexity")
```

### 4. From External Files

#### CSV File

```csv
read_id,class,score
read_1,0,0.95
read_2,1,0.88
read_3,0,0.92
```

```python
extractor = TargetExtractor.from_file(
    filepath="labels.csv",
    id_column="read_id",
    label_column="class",
    converter=int
)
```

#### JSON File

```json
[
    {"id": "read_1", "label": 0, "confidence": 0.95},
    {"id": "read_2", "label": 1, "confidence": 0.88}
]
```

```python
extractor = TargetExtractor.from_file(
    filepath="labels.json",
    id_column="id",
    label_column="label",
    converter=int
)
```

### 5. Custom Extraction

```python
def custom_extractor(record):
    """Custom logic to extract target from record."""
    header = record["id"].decode()
    sequence = record["sequence"]
    quality = record.get("quality", [])

    # Your custom logic here
    gc_content = sequence.count(b'G') + sequence.count(b'C')
    gc_ratio = gc_content / len(sequence)

    return gc_ratio

extractor = TargetExtractor(custom_extractor)
```

## Classification Tasks

### Binary Classification

```python
from deepbiop.targets import create_classification_extractor

# Automatically maps class names to indices
extractor = create_classification_extractor(
    classes=["negative", "positive"],
    pattern=r"class=(\w+)"
)

# negative -> 0, positive -> 1
```

### Multi-Class Classification

```python
extractor = create_classification_extractor(
    classes=["type_a", "type_b", "type_c", "type_d"],
    key="type"  # Extract from key:value pairs
)
```

## Complete Training Examples

### Example 1: Quality Score Prediction

```python
import torch.nn as nn
import pytorch_lightning as pl
from deepbiop.lightning import BiologicalDataModule
from deepbiop.targets import TargetExtractor
from deepbiop import OneHotEncoder

# Data
data_module = BiologicalDataModule(
    train_path="train.fastq",
    val_path="val.fastq",
    test_path="test.fastq",
    transform=OneHotEncoder(),
    target_fn=TargetExtractor.from_quality("mean"),
    collate_mode="tensor",
    batch_size=32,
)

# Model
class QualityPredictor(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv1d(4, 64, kernel_size=7),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
            nn.Linear(128, 1)
        )
        self.loss = nn.MSELoss()

    def forward(self, x):
        # x: (batch, seq_len, 4) -> transpose to (batch, 4, seq_len)
        return self.model(x.transpose(1, 2)).squeeze()

    def training_step(self, batch, batch_idx):
        features, targets = batch["features"], batch["targets"]
        preds = self(features)
        loss = self.loss(preds, targets)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

# Train
model = QualityPredictor()
trainer = pl.Trainer(max_epochs=10)
trainer.fit(model, data_module)
```

### Example 2: Read Classification

```python
from deepbiop import KmerEncoder

# Data with external labels
data_module = BiologicalDataModule(
    train_path="train.fastq",
    val_path="val.fastq",
    label_file="labels.csv",  # External label file
    transform=KmerEncoder(k=6),  # 6-mer encoding
    collate_mode="tensor",
    batch_size=64,
)

# Model
class ReadClassifier(pl.LightningModule):
    def __init__(self, input_dim=4096, num_classes=2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        features, targets = batch["features"], batch["targets"]
        logits = self(features)
        loss = self.loss(logits, targets.long())

        # Calculate accuracy
        preds = logits.argmax(dim=1)
        acc = (preds == targets).float().mean()

        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

# Train
model = ReadClassifier()
trainer = pl.Trainer(max_epochs=20)
trainer.fit(model, data_module)
```

### Example 3: Multi-Task Learning

```python
def multi_target_extractor(record):
    """Extract multiple targets for multi-task learning."""
    quality = record.get("quality", [])
    mean_quality = sum(quality) / len(quality) if quality else 0.0

    header = record["id"].decode()
    read_type = 1 if "type:good" in header else 0

    return {
        "quality": mean_quality,  # Regression
        "type": read_type,        # Classification
    }

data_module = BiologicalDataModule(
    train_path="train.fastq",
    val_path="val.fastq",
    transform=OneHotEncoder(),
    target_fn=TargetExtractor(multi_target_extractor),
    collate_mode="tensor",
    batch_size=32,
)

class MultiTaskModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(4, 64, kernel_size=7),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
        )
        # Task-specific heads
        self.quality_head = nn.Linear(64, 1)
        self.type_head = nn.Linear(64, 2)

        self.quality_loss = nn.MSELoss()
        self.type_loss = nn.CrossEntropyLoss()

    def forward(self, x):
        features = self.encoder(x.transpose(1, 2))
        return {
            "quality": self.quality_head(features).squeeze(),
            "type": self.type_head(features)
        }

    def training_step(self, batch, batch_idx):
        features = batch["features"]
        targets = batch["targets"]

        preds = self(features)

        # Extract individual targets
        quality_targets = torch.tensor([t["quality"] for t in targets])
        type_targets = torch.tensor([t["type"] for t in targets])

        # Calculate losses
        quality_loss = self.quality_loss(preds["quality"], quality_targets)
        type_loss = self.type_loss(preds["type"], type_targets.long())

        # Combined loss
        total_loss = quality_loss + type_loss

        self.log("train/quality_loss", quality_loss)
        self.log("train/type_loss", type_loss)
        self.log("train/total_loss", total_loss)

        return total_loss
```

## PyTorch DataLoader (Without Lightning)

```python
from torch.utils.data import DataLoader
from deepbiop.fq import FastqStreamDataset
from deepbiop.transforms import TransformDataset
from deepbiop.targets import TargetExtractor
from deepbiop.collate import tensor_collate
from deepbiop import OneHotEncoder

# Create dataset
base_dataset = FastqStreamDataset("train.fastq")

dataset = TransformDataset(
    base_dataset,
    transform=OneHotEncoder(),
    target_fn=TargetExtractor.from_quality("mean"),
    return_dict=False,  # Return (features, target) tuples
)

# Create DataLoader
loader = DataLoader(
    dataset,
    batch_size=32,
    collate_fn=tensor_collate,
    num_workers=4,
)

# Training loop
for features, targets in loader:
    # features: torch.Tensor of shape (batch, seq_len, 4)
    # targets: torch.Tensor of shape (batch,)

    predictions = model(features)
    loss = criterion(predictions, targets)

    loss.backward()
    optimizer.step()
```

## Collate Functions

DeepBioP provides three collate modes:

### 1. `default` - Identity Collation

Returns batch as-is (list of samples). Use for variable-length sequences.

```python
from deepbiop.collate import default_collate

loader = DataLoader(dataset, batch_size=32, collate_fn=default_collate)
# batch is a list of dicts
```

### 2. `supervised` - Structured Collation

Separates features and targets into dict keys.

```python
from deepbiop.collate import supervised_collate

loader = DataLoader(dataset, batch_size=32, collate_fn=supervised_collate)
# batch = {"features": [...], "targets": [...], "ids": [...]}
```

### 3. `tensor` - Tensor Collation

Stacks features and targets into PyTorch tensors.

```python
from deepbiop.collate import tensor_collate

loader = DataLoader(dataset, batch_size=32, collate_fn=tensor_collate)
# batch = {"features": Tensor(...), "targets": Tensor(...)}
```

## Supported File Formats

- **FASTQ** (`.fastq`, `.fq`, `.fastq.gz`): Full support including quality scores
- **FASTA** (`.fasta`, `.fa`, `.fasta.gz`): Sequence only (no quality)
- **BAM** (`.bam`): Aligned reads with metadata

## Built-in Extractors Reference

### Quality Extractors (FASTQ only)

```python
TargetExtractor.from_quality(stat="mean")    # Mean quality
TargetExtractor.from_quality(stat="median")  # Median quality
TargetExtractor.from_quality(stat="min")     # Minimum quality
TargetExtractor.from_quality(stat="max")     # Maximum quality
TargetExtractor.from_quality(stat="std")     # Standard deviation
```

### Sequence Extractors

```python
TargetExtractor.from_sequence(feature="gc_content")  # GC content (0-1)
TargetExtractor.from_sequence(feature="length")      # Sequence length
TargetExtractor.from_sequence(feature="complexity")  # Shannon entropy
```

### Convenience Functions

```python
from deepbiop.targets import get_builtin_extractor

# Get by name
extractor = get_builtin_extractor("quality_mean")
extractor = get_builtin_extractor("gc_content")
```

## Best Practices

1. **Use Built-in Extractors** when possible for efficiency
2. **Choose Appropriate Encoding**:
   - `OneHotEncoder`: For CNNs, interpretability
   - `KmerEncoder`: For capturing motifs, faster training
   - `IntegerEncoder`: For RNNs, embedding layers

3. **Collate Mode Selection**:
   - `"default"`: Variable-length, custom batching
   - `"supervised"`: Standard supervised learning
   - `"tensor"`: Maximum efficiency with fixed-length

4. **Return Format**:
   - `return_dict=True`: Keep metadata, debugging
   - `return_dict=False`: PyTorch-style tuples, cleaner training loops

5. **Memory Efficiency**:
   - Use streaming datasets for large files
   - Set appropriate `batch_size` and `num_workers`
   - Consider data augmentation for small datasets

## Troubleshooting

### "Sequence ID not found in label file"

Ensure sequence IDs in your data match those in the label file. IDs are extracted after removing `@` prefix and taking first whitespace-delimited token.

### "Pattern not found in header"

Verify your regex pattern matches your header format. Test with:

```python
import re
header = "@read_1 label=positive"
pattern = r"label=(\w+)"
match = re.search(pattern, header)
print(match.group(1) if match else "No match")
```

### Quality scores not available

FASTA files don't have quality scores. Use sequence-based extractors instead:

```python
# Won't work with FASTA
extractor = TargetExtractor.from_quality("mean")  # Error!

# Use this instead
extractor = TargetExtractor.from_sequence("gc_content")  # OK
```

### Variable-length sequences

For variable-length sequences, use `collate_mode="default"` or `collate_mode="supervised"` and handle batching manually, or use padding/truncation in your transform.

## See Also

- [examples/supervised_learning.py](supervised_learning.py) - Complete runnable examples
- [DeepBioP Documentation](https://github.com/cauliyang/DeepBioP) - Main documentation
- [PyTorch Lightning Documentation](https://lightning.ai/docs/pytorch/stable/) - Training framework
