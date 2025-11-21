# Implementation Tasks: PyTorch-Style Python API for Deep Learning

**Branch**: `002-pytorch-api` | **Date**: 2025-11-03
**Spec**: [spec.md](./spec.md) | **Plan**: [plan.md](./plan.md)

## Overview

This document provides dependency-ordered implementation tasks for the PyTorch-compatible Python API. Tasks are organized by user story (from spec.md) to enable independent, parallel implementation and testing.

**Implementation Strategy**: Test-First, Incremental Delivery
- Each user story is independently testable
- MVP = User Story 1 (P1) only
- Subsequent stories add features without breaking existing functionality
- Parallel implementation opportunities marked with [P]

**Total Tasks**: 39 tasks across 7 phases
**Estimated Effort**: 500-800 LOC wrapper code

---

## Task Format Legend

```
- [ ] T### [P] [US#] Description with file path
       │    │    │    └─ Action + exact file location
       │    │    └────── User Story label (US1, US2, etc.)
       │    └─────────── Parallelizable (different files, no blocking dependencies)
       └──────────────── Task ID (sequential execution order)
```

---

## Dependencies & Execution Order

### User Story Dependency Graph

```
Phase 1 (Setup)
    │
    ▼
Phase 2 (Foundational - File readers, encoders, augmentations already exist)
    │
    ├──────────────┬──────────────┬──────────────┬──────────────┐
    ▼              ▼              ▼              ▼              ▼
  US1 (P1)      US2 (P2)      US3 (P3)      US4 (P4)      US5 (P5)
 Dataset/       Data           Model         Export        Inspection
 DataLoader   Augmentation    Ready         Cache         Validation
    │              │              │              │              │
    └──────────────┴──────────────┴──────────────┴──────────────┘
                            │
                            ▼
                      Phase 7 (Polish)
```

**Parallelization Opportunities**:
- User Stories 1-5 are **independent** - can be implemented in parallel by different developers
- Within each story: Tasks marked [P] can run in parallel
- Foundational phase is empty (existing DeepBioP functionality is sufficient)

---

## Phase 1: Setup & Module Structure

**Goal**: Initialize Python module structure for PyTorch API

**Duration**: ~30 minutes

### Tasks

- [X] T001 Create pytorch module directory structure in py-deepbiop/src/pytorch/
- [X] T002 Create pytorch module entry point py-deepbiop/src/pytorch/mod.rs
- [X] T003 Register pytorch module in py-deepbiop/src/lib.rs
- [X] T004 Update py-deepbiop/Cargo.toml dependencies (ensure pyo3, numpy, rayon)
- [X] T005 Create stub files: py-deepbiop/src/pytorch/dataset.rs (empty), dataloader.rs, transforms.rs, collate.rs, cache.rs
- [X] T006 Add pytorch module to stub generation in py-deepbiop/build script
- [X] T007 Create test file py-deepbiop/tests/test_pytorch_api.py with structure
- [X] T008 Create example file py-deepbiop/examples/pytorch_quickstart.py (skeleton)

**Validation**: `cargo build` succeeds, pytorch module imports in Python

---

## Phase 2: Foundational Components

**Goal**: Establish shared infrastructure (most functionality already exists in DeepBioP)

**Note**: This phase is **minimal** because we're wrapping existing functionality:
- File readers: Already exist (deepbiop-fq, deepbiop-fa)
- Encoders: Already exist (OneHotEncoder, IntegerEncoder, KmerEncoder)
- Augmentations: Already exist (ReverseComplement, Mutator, Sampler)

### Tasks

- [X] T009 [P] Create common types module py-deepbiop/src/pytorch/types.rs (Sample, Batch TypedDicts)
- [X] T010 [P] Create error conversion utilities in py-deepbiop/src/pytorch/errors.rs (Rust → Python exceptions)

**Validation**: Types compile, error conversion works

---

## Phase 3: User Story 1 (P1) - Dataset Loading & Preprocessing

**User Story**: A computational biologist wants to load FASTQ sequencing data and prepare it for training, similar to PyTorch's Dataset and DataLoader.

**Goal**: Implement core Dataset and DataLoader classes with basic encoding

**Independent Test**: Load FASTQ file, apply one-hot encoding, iterate through batches

**Success Criteria** (from spec.md):
- ✅ Load and validate FASTQ files without manual parsing
- ✅ Apply encoding transformations (one-hot, k-mer, integer)
- ✅ Create batches with consistent shapes
- ✅ Iterate through batches with proper tensor format

### Tasks

#### 3.1 Dataset Implementation

- [X] T011 [US1] Write failing test: test_dataset_creation() in py-deepbiop/tests/test_pytorch_api.py
- [X] T012 [US1] Implement Dataset.__init__() in py-deepbiop/src/pytorch/dataset.rs (wrap existing FASTQ reader)
- [X] T013 [US1] Implement Dataset.__len__() using existing reader's count
- [X] T014 [US1] Implement Dataset.__getitem__() with lazy loading from existing reader
- [X] T015 [US1] Implement Dataset.__iter__() returning separate Iterator instance
- [X] T016 [US1] Verify test_dataset_creation() passes

#### 3.2 Transform Wrappers

- [X] T017 [P] [US1] Write failing test: test_onehot_encoder() in py-deepbiop/tests/test_pytorch_api.py
- [X] T018 [P] [US1] Implement OneHotEncoder wrapper in py-deepbiop/src/pytorch/transforms.rs (delegates to deepbiop-fq)
- [X] T019 [P] [US1] Write failing test: test_integer_encoder()
- [X] T020 [P] [US1] Implement IntegerEncoder wrapper (delegates to deepbiop-fq)
- [X] T021 [P] [US1] Write failing test: test_kmer_encoder()
- [X] T022 [P] [US1] Implement KmerEncoder wrapper (delegates to deepbiop-core)
- [X] T023 [US1] Verify all encoder tests pass

#### 3.3 DataLoader Implementation

- [X] T024 [US1] Write failing test: test_dataloader_batching() in py-deepbiop/tests/test_pytorch_api.py
- [X] T025 [US1] Implement DataLoader.__init__() in py-deepbiop/src/pytorch/dataloader.rs
- [X] T026 [US1] Implement DataLoader.__iter__() with shuffling support (uses Rust rand crate)
- [X] T027 [US1] Implement DataLoader.__len__() calculation
- [X] T028 [US1] Verify test_dataloader_batching() passes

#### 3.4 Collate Function

- [X] T029 [US1] Write failing test: test_default_collate() in py-deepbiop/tests/test_pytorch_api.py
- [X] T030 [US1] Implement default_collate_fn() in py-deepbiop/src/pytorch/collate.rs (uses NumPy pad/stack)
- [X] T031 [US1] Verify test_default_collate() passes

#### 3.5 Integration Test

- [X] T032 [US1] Write integration test: test_full_pipeline() (load → transform → batch → PyTorch)
- [X] T033 [US1] Update example py-deepbiop/examples/pytorch_quickstart.py with working code
- [X] T034 [US1] Verify integration test passes, example runs

**Phase 3 Validation**:
- All US1 tests pass (8 tests)
- Can load FASTQ, encode sequences, generate batches
- Batches have shape [batch_size, max_len, features]
- Example in examples/pytorch_quickstart.py demonstrates end-to-end workflow

**MVP Checkpoint**: ✅ At this point, User Story 1 is complete and deliverable as MVP

---

## Phase 4: User Story 2 (P2) - Data Augmentation Pipeline

**User Story**: A researcher wants to apply augmentations (reverse complement, mutations, sampling) using a composable pipeline like torchvision transforms.

**Goal**: Implement Compose class and augmentation wrappers

**Independent Test**: Apply chained transforms (sample → mutate → encode), verify output

**Success Criteria** (from spec.md):
- ✅ Compose multiple transformations in order
- ✅ Output maintains biological validity
- ✅ Augmentations applied on-the-fly during batch generation
- ✅ Reproducible with random seeds

### Tasks

- [X] T035 [P] [US2] Write failing test: test_compose() in py-deepbiop/tests/test_pytorch_api.py
- [X] T036 [P] [US2] Implement Compose class in py-deepbiop/src/pytorch/transforms.rs (<20 LOC)
- [X] T037 [P] [US2] Write failing test: test_reverse_complement()
- [X] T038 [P] [US2] Implement ReverseComplement wrapper (delegates to deepbiop-fq)
- [X] T039 [P] [US2] Write failing test: test_mutator()
- [X] T040 [P] [US2] Implement Mutator wrapper (delegates to deepbiop-fq)
- [X] T041 [P] [US2] Write failing test: test_sampler()
- [X] T042 [P] [US2] Implement Sampler wrapper (delegates to deepbiop-fq)
- [X] T043 [US2] Write integration test: test_augmentation_pipeline()
- [X] T044 [US2] Verify all US2 tests pass (6 tests)

**Phase 4 Validation**:
- Compose chains transforms correctly
- Augmentations preserve valid sequences
- Reproducible with seeds

---

## Phase 5: User Story 3 (P3) - Model-Ready Tensor Operations

**User Story**: A deep learning engineer wants tensor operations (indexing, slicing) that integrate with PyTorch models.

**Goal**: Ensure NumPy array outputs work seamlessly with PyTorch

**Independent Test**: Convert batch to PyTorch tensor, pass to model, verify no errors

**Success Criteria** (from spec.md):
- ✅ Indexing/slicing returns properly shaped tensors
- ✅ Concatenation maintains data integrity
- ✅ Forward pass executes without type/shape errors
- ✅ Standard PyTorch loss functions work

### Tasks

- [X] T045 [P] [US3] Write failing test: test_numpy_pytorch_conversion() in py-deepbiop/tests/test_pytorch_api.py
- [X] T046 [P] [US3] Write failing test: test_batch_indexing_slicing()
- [X] T047 [P] [US3] Write failing test: test_pytorch_model_integration()
- [X] T048 [US3] Verify NumPy arrays have correct dtype (float32) and C-order contiguous layout
- [X] T049 [US3] Verify all US3 tests pass (3 tests)

**Phase 5 Validation**:
- Zero-copy conversion with torch.from_numpy() works
- Batches integrate with PyTorch models without modification

---

## Phase 6: User Story 4 (P4) - Export & Persistence

**User Story**: A researcher wants to save processed datasets in efficient formats (Parquet, HDF5) for reuse.

**Goal**: Implement Cache class wrapping existing export functionality

**Independent Test**: Save processed dataset, reload in new session, verify 10x faster

**Success Criteria** (from spec.md):
- ✅ Save dataset with metadata
- ✅ Load quickly without re-processing
- ✅ Storage optimized through compression
- ✅ Shareable across researchers

### Tasks

- [X] T050 [P] [US4] Write failing test: test_cache_save() in py-deepbiop/tests/test_pytorch_api.py
- [X] T051 [P] [US4] Write failing test: test_cache_load()
- [X] T052 [P] [US4] Write failing test: test_cache_invalidation()
- [X] T053 [US4] Implement Cache.save() in py-deepbiop/src/pytorch/cache.rs (delegates to NumPy savez_compressed)
- [X] T054 [US4] Implement Cache.load() (delegates to NumPy load)
- [X] T055 [US4] Implement Cache.is_valid() with metadata checking (mtime)
- [X] T056 [US4] Create metadata JSON writer/reader for cache invalidation
- [X] T057 [US4] Verify all US4 tests pass (3 tests)

**Phase 6 Validation**:
- Cached datasets load 10x faster
- Cache invalidates when source files change

---

## Phase 7: User Story 5 (P5) - Dataset Inspection & Validation

**User Story**: A researcher wants to inspect dataset properties and validate data quality before training.

**Goal**: Implement Dataset.summary() using NumPy functions

**Independent Test**: Call summary(), verify statistics match expected values

**Success Criteria** (from spec.md):
- ✅ Report sample counts, length distribution, memory footprint
- ✅ Show class distribution
- ✅ Check for NaN/inf values
- ✅ Provide actionable warnings

### Tasks

- [X] T058 [P] [US5] Write failing test: test_dataset_summary() in py-deepbiop/tests/test_pytorch_api.py
- [X] T059 [P] [US5] Implement Dataset.summary() using np.unique, np.histogram (minimal custom code)
- [X] T060 [P] [US5] Write failing test: test_dataset_validation()
- [X] T061 [P] [US5] Implement validation in Dataset (delegates to existing deepbiop validation)
- [X] T062 [US5] Verify all US5 tests pass (2 tests)

**Phase 7 Validation**:
- Summary completes in <1s per 10k sequences
- Validation catches common data issues

---

## Phase 8: Polish & Cross-Cutting Concerns

**Goal**: Finalize documentation, performance optimization, and packaging

### Tasks

#### 8.1 Documentation

- [X] T063 [P] Generate Python type stubs via cargo run --bin stub_gen (skipped - auto-generated)
- [X] T064 [P] Update py-deepbiop/README.md with PyTorch API usage
- [X] T065 [P] Verify all docstrings match api-contracts.md signatures

#### 8.2 Performance Validation

- [X] T066 [P] Run benchmark: test_batch_generation_throughput() (target: 10k+ seq/s)
- [X] T067 [P] Run benchmark: test_cache_speedup() (target: 10x faster)
- [X] T068 [P] Profile GIL release in batch operations (verify py.detach() working)

#### 8.3 Integration & Packaging

- [X] T069 Verify py-deepbiop builds with maturin develop
- [X] T070 Run full test suite: pytest py-deepbiop/tests/test_pytorch_api.py
- [X] T071 Verify examples/pytorch_quickstart.py runs successfully
- [X] T072 Update CHANGELOG.md with PyTorch API feature

**Phase 8 Validation**:
- All 22+ tests pass
- Performance targets met (SC-002, SC-005, SC-006, SC-007)
- Type stubs generated
- Example code runs

---

## Parallel Execution Examples

### MVP Implementation (User Story 1 Only)

**Week 1: Foundation**
```bash
# Developer A
T001-T008 (Setup)
T009-T010 (Types & errors)
T011-T016 (Dataset)

# Developer B (can start after T008)
T017-T023 (Transform wrappers)
```

**Week 2: DataLoader & Integration**
```bash
# Developer A
T024-T028 (DataLoader)

# Developer B
T029-T031 (Collate)

# Both
T032-T034 (Integration test & example)
```

### Full Feature Implementation (All User Stories)

**After MVP (Stories 2-5 can be parallel)**

```bash
# Team Member 1
US2 (T035-T044): Augmentation pipeline

# Team Member 2
US3 (T045-T049): PyTorch integration

# Team Member 3
US4 (T050-T057): Cache implementation

# Team Member 4
US5 (T058-T062): Inspection & validation

# All
Phase 8 (T063-T072): Polish (can be distributed)
```

**Timeline Estimate**:
- MVP (US1): 1-2 weeks (1 developer) or 3-5 days (2 developers parallel)
- Full feature: 2-3 weeks (4 developers parallel) or 4-6 weeks (1 developer sequential)

---

## Task Summary by Phase

| Phase | User Story | Task Count | Parallel Tasks | Key Deliverable |
|-------|------------|------------|----------------|-----------------|
| 1 | Setup | 8 | 0 | Module structure |
| 2 | Foundation | 2 | 2 | Types & errors |
| 3 | US1 (P1) | 24 | 6 | Dataset + DataLoader |
| 4 | US2 (P2) | 10 | 8 | Augmentation pipeline |
| 5 | US3 (P3) | 5 | 3 | PyTorch compatibility |
| 6 | US4 (P4) | 8 | 3 | Cache/export |
| 7 | US5 (P5) | 5 | 4 | Inspection/validation |
| 8 | Polish | 10 | 8 | Docs, benchmarks |
| **Total** | - | **72** | **34** (47%)| Complete feature |

---

## MVP Definition

**Minimum Viable Product** = User Story 1 (P1) **ONLY**

**What's Included**:
- Dataset class (load FASTQ/FASTA files)
- DataLoader class (batching, shuffling)
- Transform wrappers (OneHotEncoder, IntegerEncoder, KmerEncoder)
- Default collate function
- Basic example (examples/pytorch_quickstart.py)

**What's Excluded** (add later):
- Augmentation pipeline (US2)
- Advanced PyTorch integration tests (US3)
- Caching (US4)
- Inspection/validation (US5)

**Value**: Researchers can immediately use DeepBioP for deep learning with familiar PyTorch patterns

**Tasks for MVP**: T001-T034 (34 tasks)
**Estimated Effort**: 1-2 weeks (1 developer) or 3-5 days (2 developers)

---

## Implementation Notes

### Test-First Workflow

For each task group:
1. Write failing test first (Red)
2. Implement minimum code to pass (Green)
3. Refactor while keeping tests green
4. Move to next task

### Code Reuse Checklist

Before implementing any task, verify:
- [ ] Does existing DeepBioP functionality cover this?
- [ ] Can I delegate to existing crate instead of duplicating?
- [ ] Is this a thin wrapper (<20 LOC for Compose, <50 LOC for collate)?

### GIL Release Pattern

For any batch operation using Rayon:
```rust
let encoded = py
    .detach(|| self.inner.encode_batch(&seq_refs))
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
```

### File Organization

```
py-deepbiop/src/pytorch/
├── mod.rs              # Module registration (T002)
├── types.rs            # Sample, Batch types (T009)
├── errors.rs           # Error conversion (T010)
├── dataset.rs          # Dataset class (T012-T016)
├── dataloader.rs       # DataLoader class (T025-T027)
├── transforms.rs       # Transform wrappers (T018-T022, T036-T042)
├── collate.rs          # Collate functions (T030)
└── cache.rs            # Cache implementation (T053-T056)
```

---

## Next Steps

1. **Start with MVP**: Execute T001-T034 for User Story 1
2. **Validate MVP**: Run integration test, verify example works
3. **Iterate**: Add User Stories 2-5 based on user feedback
4. **Optimize**: Profile and optimize bottlenecks (Phase 8)

**Ready to implement?** Start with Phase 1, Task T001.
