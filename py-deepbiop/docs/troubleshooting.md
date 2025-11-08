# Troubleshooting Guide

Common errors and solutions when using DeepBioP.

## Installation Issues

### Import Error: No module named 'deepbiop'

**Problem:**
```python
>>> import deepbiop
ModuleNotFoundError: No module named 'deepbiop'
```

**Solutions:**
1. Install DeepBioP: `pip install deepbiop` or `uv pip install deepbiop`
2. Check Python environment: `which python` and `pip list | grep deepbiop`
3. Try in a fresh virtual environment

---

### ImportError: Cannot import from deepbiop.fq

**Problem:**
```python
from deepbiop.fq import FastqStreamDataset  # ModuleNotFoundError
```

**Solution:**
Use the correct import style:
```python
# Correct
from deepbiop import fq
dataset = fq.FastqStreamDataset("file.fastq")

# Or
import deepbiop
dataset = deepbiop.fq.FastqStreamDataset("file.fastq")
```

---

## Data Loading Issues

### File Not Found Error

**Problem:**
```
FileNotFoundError: No such file or directory: 'data.fastq'
```

**Solutions:**
1. Use absolute paths: `/full/path/to/data.fastq`
2. Check current directory: `import os; print(os.getcwd())`
3. Verify file exists: `from pathlib import Path; Path('data.fastq').exists()`

---

### Gzipped File Not Recognized

**Problem:**
```
Error reading gzipped FASTQ file
```

**Solutions:**
1. Use correct extensions: `.fastq.gz`, `.fq.gz`, `.fastq.bgz`
2. Verify file is actually gzipped: `file data.fastq.gz`
3. For bgzip files, ensure they're properly indexed

---

## Transform Issues

### TypeError: QualityFilter got unexpected keyword argument 'min_quality'

**Problem:**
```python
filter = QualityFilter(min_quality=30.0)  # TypeError
```

**Solution:**
Use correct parameter name:
```python
# Correct parameter name
filter = fq.QualityFilter(min_mean_quality=30.0)
```

Check API documentation for correct parameter names.

---

### Transforms Not Working with Datasets

**Problem:**
Transforms don't seem to apply to dataset records.

**Solution:**
Use `TransformDataset` wrapper:
```python
from deepbiop.transforms import TransformDataset, Compose
from deepbiop import fq

dataset = fq.FastqStreamDataset("data.fastq")

# Wrap with transforms
processed = TransformDataset(
    dataset,
    transform=Compose([
        fq.Mutator(mutation_rate=0.1),
        fq.OneHotEncoder()
    ])
)

for record in processed:  # Now transforms are applied
    pass
```

---

## PyTorch Integration Issues

### DataLoader Hangs with num_workers > 0

**Problem:**
DataLoader freezes when using multiple workers.

**Solutions:**
1. Start with `num_workers=0` to test
2. On macOS, use `num_workers=0` or set `OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES`
3. Ensure dataset supports pickling (DeepBioP datasets do)
4. Use `persistent_workers=True` for Lightning

```python
# Workaround for macOS
import os
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'

loader = DataLoader(dataset, num_workers=4)
```

---

### Sequences Have Different Lengths

**Problem:**
```
RuntimeError: stack expects each tensor to be equal size
```

**Solution:**
Use custom collate function with padding:
```python
import torch
from torch.utils.data import DataLoader

def collate_fn(batch):
    sequences = [torch.from_numpy(item['sequence']) for item in batch]
    max_len = max(seq.shape[0] for seq in sequences)

    padded = torch.zeros(len(sequences), max_len, dtype=sequences[0].dtype)
    for i, seq in enumerate(sequences):
        padded[i, :seq.shape[0]] = seq

    return {'sequences': padded}

loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)
```

---

## Lightning Integration Issues

### BiologicalDataModule: Dataset Not Set Up

**Problem:**
```
RuntimeError: Train dataset not set up. Call setup(stage='fit') first.
```

**Solution:**
Call `setup()` before getting dataloaders:
```python
dm = BiologicalDataModule(train_path="train.fastq", batch_size=32)
dm.setup(stage='fit')  # Must call setup first
train_loader = dm.train_dataloader()
```

Or use with Lightning Trainer (calls setup automatically):
```python
trainer = pl.Trainer()
trainer.fit(model, dm)  # Trainer calls setup() for you
```

---

## Performance Issues

### Slow Data Loading

**Solutions:**
1. **Use multiple workers:**
   ```python
   loader = DataLoader(dataset, num_workers=4)
   ```

2. **Enable pin_memory for GPU:**
   ```python
   loader = DataLoader(dataset, pin_memory=True)
   ```

3. **Use streaming datasets** for large files:
   ```python
   # FastqStreamDataset instead of FastqDataset
   dataset = fq.FastqStreamDataset("large.fastq.gz")
   ```

4. **Profile bottlenecks:**
   ```python
   import time
   start = time.time()
   for i, batch in enumerate(loader):
       if i >= 100: break
   print(f"Time per batch: {(time.time()-start)/100:.3f}s")
   ```

---

### High Memory Usage

**Solutions:**
1. **Reduce batch size:**
   ```python
   loader = DataLoader(dataset, batch_size=16)  # Instead of 128
   ```

2. **Use streaming datasets:**
   ```python
   # Constant memory usage
   dataset = fq.FastqStreamDataset("file.fastq.gz")
   ```

3. **Process in chunks:**
   ```python
   # Don't load entire file into memory
   for record in dataset:
       process(record)  # Process one at a time
   ```

4. **Monitor memory:**
   ```python
   import psutil
   import os

   process = psutil.Process(os.getpid())
   print(f"Memory: {process.memory_info().rss / 1024 / 1024:.0f} MB")
   ```

---

## Common Gotchas

### Streaming Datasets Don't Support Indexing

**Problem:**
```python
dataset = fq.FastqStreamDataset("file.fastq")
record = dataset[0]  # TypeError
```

**Solution:**
Use iterator or switch to random-access dataset:
```python
# Option 1: Use iterator
record = next(iter(dataset))

# Option 2: Use FastqDataset for indexing
dataset = fq.FastqDataset("file.fastq")
record = dataset[0]  # Works!
```

---

### Transforms Modify Original Records

**Problem:**
Transforms modify the original record dict.

**Solution:**
This is intentional for memory efficiency. If you need the original:
```python
import copy

for record in dataset:
    original = copy.deepcopy(record)
    transformed = transform(record)
```

---

## Error Messages

### "Unknown base in sequence"

**Problem:**
Sequence contains non-ACGT bases.

**Solution:**
Use appropriate `unknown_strategy`:
```python
encoder = fq.OneHotEncoder(
    encoding_type="dna",
    unknown_strategy="skip"  # or "mask" or "error"
)
```

---

### "Quality scores length doesn't match sequence"

**Problem:**
Malformed FASTQ file.

**Solution:**
1. Validate FASTQ file format
2. Check for truncated files
3. Use error handling:
   ```python
   try:
       for record in dataset:
           process(record)
   except Exception as e:
       print(f"Error at record {record.get('id', 'unknown')}: {e}")
   ```

---

## Getting Help

If you encounter an issue not covered here:

1. **Search existing issues**: https://github.com/cauliyang/DeepBioP/issues
2. **Check documentation**: [API Reference](api-reference.md) | [Quick Start](quickstart.md)
3. **Create new issue**: Include:
   - DeepBioP version: `pip show deepbiop`
   - Python version: `python --version`
   - Minimal reproducible example
   - Full error traceback

4. **Discussions**: https://github.com/cauliyang/DeepBioP/discussions

---

**Last Updated**: 2025-11-07
