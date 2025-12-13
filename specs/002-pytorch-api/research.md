# Research: PyTorch-Style Python API Technical Decisions

**Date**: 2025-11-03
**Feature**: PyTorch-Style Python API for Deep Learning
**Phase**: 0 - Research & Technical Resolution

This document consolidates research findings that resolve all technical unknowns identified in the planning phase. All decisions prioritize code reuse, performance, and PyTorch API compatibility.

---

## Decision 1: PyO3 Iterator Pattern for Dataset/DataLoader

### Decision

Use **separate Dataset (indexable) and Iterator (stateful) classes** with PyO3 0.27 best practices:
- Dataset implements `__len__`, `__getitem__` (Map-Style)
- Dataset returns separate Iterator instance from `__iter__`
- Iterator implements `__iter__` (returns self) and `__next__` (returns `Option<T>`)
- Annotate Dataset with `#[pyclass(sequence)]` for PyTorch/NumPy compatibility

### Rationale

1. **PyTorch compatibility**: Matches torch.utils.data.Dataset protocol exactly
2. **Separation of concerns**: Stateless indexing (Dataset) vs stateful iteration (Iterator)
3. **Multiple independent iterators**: Each epoch can create fresh iterator instance
4. **Existing pattern**: Already used in `crates/deepbiop-fq/src/dataset.rs` (lines 46-342)

### Implementation Pattern

```rust
use pyo3::prelude::*;
use pyo3_stub_gen::derive::*;

#[gen_stub_pyclass]
#[pyclass(name = "BioDataset", module = "deepbiop.pytorch", sequence)]
pub struct BioDataset {
    file_path: String,
    records_count: usize,
}

#[gen_stub_pymethods]
#[pymethods]
impl BioDataset {
    fn __len__(&self) -> PyResult<usize> {
        Ok(self.records_count)
    }

    fn __getitem__(&self, idx: usize, py: Python) -> PyResult<Py<PyAny>> {
        // Random access to sample at index idx
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyResult<Py<BioIterator>> {
        Python::attach(|py| {
            let dataset_clone = Py::new(py, slf.clone())?;
            Py::new(py, BioIterator {
                dataset: dataset_clone,
                current_idx: 0
            })
        })
    }
}

#[gen_stub_pyclass]
#[pyclass(name = "BioIterator", module = "deepbiop.pytorch")]
pub struct BioIterator {
    dataset: Py<BioDataset>,
    current_idx: usize,
}

#[gen_stub_pymethods]
#[pymethods]
impl BioIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> PyResult<Option<Py<PyAny>>> {
        Python::attach(|py| {
            let dataset = slf.dataset.borrow(py);
            if slf.current_idx >= dataset.__len__()? {
                return Ok(None); // StopIteration
            }
            let item = dataset.__getitem__(slf.current_idx, py)?;
            slf.current_idx += 1;
            Ok(Some(item))
        })
    }
}
```

### Alternatives Considered

- **Combined Iterator-Dataset**: Rejected due to state mixing and inability to create multiple independent iterators
- **`Python::with_gil` instead of `Python::attach`**: Rejected as deprecated in PyO3 0.26+
- **No `#[pyclass(sequence)]` annotation**: Rejected as PyTorch/NumPy may not recognize sequence protocol

### References

- PyO3 Parallelism Guide: https://pyo3.rs/main/parallelism
- PyO3 Class Protocols: https://pyo3.rs/main/class/protocols
- Working implementation: `crates/deepbiop-fq/src/dataset.rs`

---

## Decision 2: GIL Release Pattern for Rayon Parallel Operations

### Decision

**Always use `py.detach()` before Rayon parallel operations** to release the GIL:

```rust
fn encode_batch<'py>(
    &self,
    py: Python<'py>,
    sequences: Vec<Vec<u8>>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let seq_refs: Vec<&[u8]> = sequences.iter().map(|s| s.as_slice()).collect();

    // Release GIL for parallel processing with Rayon
    let encoded = py
        .detach(|| self.inner.encode_batch(&seq_refs))
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    Ok(PyArray2::from_array(py, &encoded))
}
```

### Rationale

1. **Avoid deadlocks**: Without `detach()`, Rayon workers deadlock when trying to acquire GIL while main thread holds it
2. **True parallelism**: GIL release enables actual parallel execution across CPU cores
3. **Existing pattern**: Already implemented in 4 files after recent optimization work:
   - `crates/deepbiop-fq/src/augment/python.rs` (lines 39-42, 86-89)
   - `crates/deepbiop-core/src/kmer/encode.rs` (lines 438-441)
   - `crates/deepbiop-fq/src/encode/integer.rs` (lines 314-316)
   - `crates/deepbiop-fq/src/encode/onehot.rs` (lines 473-487)

### Pattern for Sharing Python Objects Across Rayon Threads

When Rayon workers need access to Python objects:

```rust
Python::attach(|outer_py| {
    // Create Python objects and wrap in Py<T> (thread-safe)
    let instances: Vec<Py<Transform>> = transforms.iter()
        .map(|t| Py::new(outer_py, t.clone()).unwrap())
        .collect();

    // Release GIL before spawning Rayon workers
    outer_py.detach(|| {
        instances.par_iter().map(|instance| {
            // Each worker attaches to get its own Python token
            Python::attach(|inner_py| {
                instance.borrow(inner_py).apply(data)
            })
        }).collect()
    })
});
```

### Alternatives Considered

- **Not releasing GIL**: Rejected due to deadlock risk and no parallelism benefit
- **`allow_threads` instead of `detach`**: Rejected as deprecated (renamed to `detach` in PyO3 0.27)

### References

- PyO3 0.27 Migration Guide: https://pyo3.rs/v0.27.1/migration.html
- GIL release examples in existing codebase (4 files listed above)

---

## Decision 3: NumPy Array Conversion Pattern

### Decision

**Use `PyArray::from_array(py, &array)` for all ndarray → NumPy conversions** (with copy):

```rust
use numpy::{PyArray2, PyArray3};
use ndarray::{Array2, Array3};

fn encode<'py>(
    &self,
    py: Python<'py>,
    sequence: Vec<u8>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    // Encoder creates owned Array2<f32>
    let encoded: Array2<f32> = self.inner.encode(&sequence)?;

    // Copy to Python memory (trivial cost compared to encoding)
    Ok(PyArray2::from_array(py, &encoded))
}
```

### Rationale

1. **Encoding cost dominates**: The copy overhead is negligible compared to sequence encoding/augmentation computation
2. **Full NumPy compatibility**: Copied array has C-order contiguous layout and supports all NumPy operations
3. **Simplicity and safety**: No lifetime complications, Python has full ownership
4. **Existing pattern**: All current encoders use this pattern consistently:
   - `crates/deepbiop-fq/src/encode/onehot.rs` (lines 452-486)
   - `crates/deepbiop-fq/src/encode/integer.rs` (lines 280-318)
   - `crates/deepbiop-core/src/kmer/encode.rs` (lines 407-443)

### Memory Ownership Flow

1. Rust encoder creates `Array2<f32>` on Rust heap (owned)
2. `PyArray2::from_array(py, &encoded)` allocates new memory in Python heap via NumPy C API
3. Data is copied from Rust array to Python array
4. Rust `Array2` is dropped when function returns (automatic cleanup)
5. Python owns the NumPy array with full control

### Alternatives Considered

- **Zero-copy with `from_owned_array()`**: Rejected as it requires restructuring encoders and limits NumPy operations
- **Pre-allocate Python array**: Rejected due to complexity and loss of Rayon parallelism benefits
- **`ToPyArray` trait**: Rejected as functionally identical to `from_array()` but less explicit

### References

- rust-numpy documentation: https://pyo3.github.io/rust-numpy/
- Zero-copy discussion: https://github.com/PyO3/rust-numpy/discussions/432
- Existing implementations across 3 encoder files

---

## Decision 4: PyTorch Dataset/DataLoader Compatibility

### Decision

**Implement Map-Style Dataset (primary) with optional Iterable-Style Dataset** for streaming:

#### Map-Style Dataset (Primary)
- Implements `__len__() -> int` and `__getitem__(idx: int) -> Sample`
- Supports random access, shuffling, and known dataset size
- Best for: Most biological datasets, files that fit in memory or have efficient indexing

#### Iterable-Style Dataset (Optional/Future)
- Implements `__iter__() -> Iterator[Sample]`
- No `__len__` required
- Best for: Streaming TB-scale genomics, cloud storage, unknown sizes

### DataLoader Essential Parameters

```python
DataLoader(
    dataset,                      # Required: Dataset instance
    batch_size: int = 1,         # Samples per batch
    shuffle: bool = False,       # Randomize data order (Map-Style only)
    num_workers: int = 0,        # Parallel data loading processes
    collate_fn: Callable = None, # Function to merge samples into batch
    drop_last: bool = False,     # Drop incomplete final batch
    pin_memory: bool = False,    # For GPU training optimization
)
```

### Collate Function Signature

```python
def collate_fn(batch: List[Sample]) -> Batch:
    """
    Merges list of samples from __getitem__() into a batch.

    Default behavior (torch.utils.data.default_collate):
    - torch.Tensor → Stack with batch dimension at index 0
    - numpy.ndarray → Convert to Tensor, then stack
    - tuple/list → Recursively collate each element
    - dict → Recursively collate values, preserve keys
    """
    pass
```

### Implementation Strategy

**For DeepBioP:**
1. Start with Map-Style Dataset (current `FastqDataset` pattern)
2. Ensure numpy array outputs for zero-copy PyTorch conversion via `torch.from_numpy()`
3. Implement custom collate functions for variable-length sequences (padding/truncation)
4. Add Iterable-Style variant later if streaming use cases emerge

### Compatibility Validation

The existing DeepBioP code already demonstrates good PyTorch compatibility:

```python
# From examples/pytorch_integration.py
class DNADataset(Dataset):
    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        encoded = self.encoder.encode(self.sequences[idx])
        X = torch.from_numpy(encoded)  # Zero-copy
        y = torch.tensor(self.labels[idx])
        return X, y

# Works with DataLoader out of the box
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
```

### Alternatives Considered

- **Iterable-Style only**: Rejected as it doesn't support shuffling and random sampling
- **Custom batch loading**: Rejected in favor of delegating to PyTorch's DataLoader

### References

- PyTorch Data API: https://docs.pytorch.org/stable/data.html
- Dataset Tutorial: https://docs.pytorch.org/tutorials/beginner/data_loading_tutorial.html
- Collate source code: https://github.com/pytorch/pytorch/blob/main/torch/utils/data/_utils/collate.py

---

## Decision 5: Module Structure and Organization

### Decision

**Implement as submodule within existing `py-deepbiop` package** (`py-deepbiop/src/pytorch/`):

```
py-deepbiop/
├── src/
│   ├── lib.rs                    # Main PyO3 module (existing)
│   └── pytorch/                  # NEW: PyTorch-style API module
│       ├── mod.rs                # Module registration
│       ├── dataset.rs            # Dataset class
│       ├── dataloader.rs         # DataLoader class
│       ├── transforms.rs         # Transform wrappers
│       ├── collate.rs            # Collate functions
│       └── cache.rs              # Cache layer
```

### Rationale

1. **Simplicity**: No new crate to maintain, faster iteration
2. **Cohesion**: Keep Python-facing code together
3. **Size estimate**: ~500-800 LOC fits well in submodule
4. **Migration path**: Can extract to separate crate later if needed (>1000 LOC or standalone Rust usage)

### Alternative Considered

- **Separate `deepbiop-pytorch` crate**: Rejected for now due to added complexity; revisit if module grows beyond 1000 LOC

---

## Decision 6: Type Hints and Stub Generation

### Decision

**Use `pyo3-stub-gen` with `#[gen_stub_pyclass]` and `#[gen_stub_pymethods]` annotations**:

```rust
use pyo3_stub_gen::derive::*;

#[gen_stub_pyclass(module = "deepbiop.pytorch")]
#[pyclass(name = "Dataset")]
pub struct Dataset {
    // ...
}

#[gen_stub_pymethods]
#[pymethods]
impl Dataset {
    #[new]
    fn new(file_path: String) -> PyResult<Self> { ... }

    fn __len__(&self) -> usize { ... }
}
```

### Rationale

1. **IDE support**: Auto-generated `.pyi` files enable autocomplete and type checking
2. **Existing infrastructure**: Already used across DeepBioP codebase
3. **Automatic sync**: Stubs regenerate on every build via `cargo run --bin stub_gen`

### References

- Existing stub files: `py-deepbiop/deepbiop/*.pyi`
- Stub generation in build: `py-deepbiop/Makefile` (line 8: `cargo run --bin stub_gen`)

---

## Decision 7: Error Handling and User Experience

### Decision

**Convert Rust errors to idiomatic Python exceptions** with actionable messages:

```rust
fn __getitem__(&self, idx: usize) -> PyResult<Sample> {
    if idx >= self.__len__()? {
        return Err(pyo3::exceptions::PyIndexError::new_err(
            format!("Index {} out of range (dataset size: {})", idx, self.__len__()?)
        ));
    }

    self.load_sample(idx)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(
            format!("Failed to load sample {}: {}", idx, e)
        ))
}
```

### Error Categories

1. **User errors** (invalid input):
   - `PyFileNotFoundError`: Missing FASTQ/FASTA files
   - `PyValueError`: Invalid parameters (batch_size=0, mutation_rate > 1.0)
   - `PyIndexError`: Out-of-bounds access

2. **Validation errors** (data format):
   - `PyValueError`: Invalid nucleotides, corrupted files
   - Include context: position, expected alphabet

3. **System errors** (I/O, resources):
   - `PyIOError`: File read failures, permission issues
   - `PyMemoryError`: OOM during large operations

### References

- PyO3 exception types: https://pyo3.rs/main/exception
- Existing error handling: `crates/deepbiop-core/src/error.rs`

---

## Summary: All Technical Unknowns Resolved

✅ **PyO3 iterator patterns** → Separate Dataset/Iterator classes with `#[pyclass(sequence)]`
✅ **GIL release** → Always use `py.detach()` before Rayon operations
✅ **NumPy conversion** → `PyArray::from_array()` with copy (optimal for current use case)
✅ **PyTorch compatibility** → Map-Style Dataset with `__len__`/`__getitem__`
✅ **Module structure** → `py-deepbiop/src/pytorch/` submodule
✅ **Type hints** → `pyo3-stub-gen` with `#[gen_stub_*]` annotations
✅ **Error handling** → Convert Rust errors to Python exceptions with context

All decisions maximize code reuse, leverage existing DeepBioP patterns, and ensure PyTorch API compatibility.

**Next Phase**: Generate design artifacts (data-model.md, contracts/, quickstart.md)
