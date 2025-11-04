# Implementation Plan: PyTorch-Style Python API for Deep Learning

**Branch**: `002-pytorch-api` | **Date**: 2025-11-03 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/002-pytorch-api/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

This feature provides a PyTorch-compatible Python API for biological sequence data loading and preprocessing, enabling researchers to use familiar DataLoader/Dataset patterns with FASTQ/FASTA files. The implementation leverages existing DeepBioP Rust functionality through thin Python wrappers (PyO3 bindings), maximizing code reuse and performance while providing an intuitive, user-friendly API that mimics PyTorch conventions.

**Key Technical Approach**:
- **Rust Layer**: New Python-facing module in a dedicated crate (`deepbiop-pytorch` or within `py-deepbiop`) using PyO3 for bindings
- **Wrapper Pattern**: All functionality delegates to existing DeepBioP crates (encoders, augmentations, file readers)
- **Parallel Processing**: Rayon for batch operations with proper GIL release (py.detach())
- **Standard Libraries**: NumPy for array operations, minimal custom code
- **Zero Duplication**: No reimplementation of existing encoder/augmentation/I/O logic

## Technical Context

**Language/Version**: Python ≥ 3.10 (abi3), Rust 1.90.0 (MSRV)
**Primary Dependencies**:
- Rust: `pyo3 ^0.27`, `pyo3-stub-gen`, `numpy` (crate), `rayon`, existing DeepBioP crates
- Python: `numpy`, `torch` (optional, for tensor compatibility), standard library only

**Storage**: N/A (delegates to existing file readers, cache uses existing Parquet/HDF5 export)
**Testing**:
- Rust: `#[cfg(test)]` inline tests + integration tests in `tests/`
- Python: `pytest` in `py-deepbiop/tests/test_pytorch_api.py`
- Contract tests: Verify PyTorch tensor compatibility, NumPy interop

**Target Platform**: Linux, macOS, Windows (via CI matrix)
**Project Type**: Python library extension (PyO3 bindings to Rust)
**Performance Goals**:
- Batch generation: 10,000+ sequences/second
- Parallel scaling: Linear up to 16 cores (Rayon with GIL release)
- Memory: Streaming I/O, peak memory ≤ 2× batch size
- Startup latency: Import overhead < 100ms

**Constraints**:
- Zero code duplication (delegate to existing DeepBioP implementations)
- PyTorch API compatibility (Dataset/DataLoader conventions)
- Minimal custom code (wrapper/adapter pattern only)
- Backward compatibility with existing DeepBioP functionality

**Scale/Scope**:
- ~5 main classes (Dataset, DataLoader, Transform, Compose, collate_fn)
- ~10 transform wrappers (encoders + augmentations)
- ~500-800 lines of Rust wrapper code
- ~200-300 lines of Python helper code (if needed)

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### ✅ I. Library-First Architecture

**Status**: PASS

**Justification**: This feature extends the existing PyO3 bindings architecture. The implementation will:
- Be self-contained within Python bindings structure (`py-deepbiop/src/pytorch.rs` or new `deepbiop-pytorch` crate)
- Depend only on existing DeepBioP crates (no new external data dependencies)
- Use feature gates if implemented as separate crate (`python-pytorch` feature)
- Be independently testable through Python tests

**Crate Decision**: Two options (to be finalized in Phase 0 research):
1. Add `pytorch` module to `py-deepbiop/src/` (simpler, recommended)
2. Create new `crates/deepbiop-pytorch` with `python` feature (more modular if functionality grows)

### ✅ II. Multi-Interface Consistency

**Status**: PASS

**Justification**: This is a Python-only API enhancement. Core logic (encoding, augmentation, file I/O) already exists in Rust and is shared via delegation. The PyTorch-style API is intentionally Python-specific to match user expectations in that ecosystem. Rust users continue using library crates directly; CLI users use existing commands.

**Interface Mapping**:
- Rust library: Existing encoders/augmentations/readers (unchanged)
- Python bindings: **NEW** PyTorch-style API (this feature) + existing low-level bindings
- CLI: No new CLI commands (out of scope per spec)

### ✅ III. Test-First Development

**Status**: PASS - Will be enforced during implementation

**Test Plan**:
1. **Python Unit Tests** (`py-deepbiop/tests/test_pytorch_api.py`):
   - Dataset creation from FASTQ/FASTA files
   - Transform application (encoding, augmentation)
   - DataLoader batching with shuffling
   - Collate function behavior (padding, stacking)
   - Cache hit/miss scenarios

2. **Python Integration Tests**:
   - End-to-end workflow: load → transform → batch → PyTorch model
   - Multi-worker DataLoader (parallel loading)
   - Large file streaming (memory constraints)

3. **Contract Tests**:
   - NumPy array outputs match expected shapes/dtypes
   - PyTorch tensor conversion (if torch available)
   - Reproducibility with random seeds

4. **Performance Benchmarks**:
   - Batch generation throughput (sequences/sec)
   - Memory usage vs batch size
   - GIL release effectiveness (parallel scaling)

**Test Data**: Reuse existing test FASTQ/FASTA files from `py-deepbiop/tests/data/`

### ✅ IV. Performance as a Feature

**Status**: PASS

**Performance Strategy**:
- **Rayon Parallelism**: Batch encoding/augmentation uses existing parallel implementations with `py.detach()` for GIL release
- **Zero-Copy**: Leverage existing `noodles` parsers, minimize data copying
- **Streaming I/O**: Dataset uses lazy loading (FR-002), no full-file buffering
- **Cache Optimization**: Delegates to existing Parquet export (compressed, columnar format)

**Performance Targets** (from spec):
- SC-002: Data loading overhead < 10% of training time
- SC-005: Process 10,000+ sequences/second with augmentation
- SC-006: Inspection completes in < 1s per 10k sequences

**Benchmarking Plan**: Add benchmarks in `py-deepbiop/benches/` if needed, or use pytest timing decorators

### ✅ V. Error Handling & User Experience

**Status**: PASS

**Error Strategy**:
- **Rust Errors**: Convert existing `DPError` types to Python exceptions via PyO3
- **Validation**: Delegate to existing sequence validation, add context (file path, sample index)
- **User Errors**: Clear messages for common mistakes (e.g., "File 'data.fq' not found", "Invalid batch_size=0")
- **Internal Errors**: Request bug report with stack trace

**Examples**:
```python
# User error - clear message
try:
    dataset = Dataset("missing.fq")
except FileNotFoundError as e:
    # "FASTQ file 'missing.fq' not found. Check path and permissions."

# Validation error - actionable
try:
    encoded = encoder.encode(b"ACGTN")
except ValueError as e:
    # "Invalid nucleotide 'N' at position 4 in sequence. Expected A, C, G, T for DNA encoding."
```

**Logging**: Use Python's `logging` module for debug output (optional, via `DEEPBIOP_LOG` env var)

### No Violations Requiring Complexity Tracking

All constitution principles are satisfied without exceptions.

## Project Structure

### Documentation (this feature)

```text
specs/002-pytorch-api/
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output (/speckit.plan command)
├── data-model.md        # Phase 1 output (/speckit.plan command)
├── quickstart.md        # Phase 1 output (/speckit.plan command)
├── contracts/           # Phase 1 output (/speckit.plan command)
│   └── api-contracts.md # Python API signatures, type annotations
└── tasks.md             # Phase 2 output (/speckit.tasks command - NOT created by /speckit.plan)
```

### Source Code (repository root)

```text
py-deepbiop/
├── src/
│   ├── lib.rs                    # Main PyO3 module (existing)
│   └── pytorch/                  # NEW: PyTorch-style API module
│       ├── mod.rs                # Module registration
│       ├── dataset.rs            # Dataset class (wraps file readers)
│       ├── dataloader.rs         # DataLoader class (batching, shuffling)
│       ├── transforms.rs         # Transform wrappers (encoders, augmentations)
│       ├── collate.rs            # Collate functions (padding, stacking)
│       └── cache.rs              # Cache layer (wraps Parquet export)
│
├── tests/
│   └── test_pytorch_api.py       # NEW: PyTorch API tests
│
└── examples/
    └── pytorch_quickstart.py     # NEW: Example usage

# Alternative structure (if separate crate chosen):
crates/deepbiop-pytorch/
├── src/
│   ├── lib.rs                    # Core Rust logic (delegates to other crates)
│   └── python.rs                 # #[cfg(feature = "python")] PyO3 bindings
├── Cargo.toml                    # Features: default = [], python = ["dep:pyo3", ...]
└── tests/
    └── integration.rs            # Rust-level integration tests

py-deepbiop/
├── src/
│   └── lib.rs                    # Import and register pytorch module
└── Cargo.toml                    # Add deepbiop-pytorch dependency with python feature
```

**Structure Decision**:

**Recommended: Option 1 (py-deepbiop/src/pytorch/ module)**
- **Why**: Simpler, fewer crates, faster iteration, no new crate to maintain
- **When to switch to Option 2**: If PyTorch API grows beyond 1000 LOC or needs standalone Rust usage

The implementation will add a `pytorch` submodule to the existing `py-deepbiop` Python package. This keeps related Python bindings together and avoids creating a new crate for what is primarily a Python-facing convenience layer. All core logic delegates to existing `deepbiop-*` crates.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

N/A - No violations

---

## Post-Design Constitution Re-Evaluation

*Re-checked after Phase 1 design artifacts completion*

### ✅ I. Library-First Architecture - CONFIRMED PASS

**Design Validation**:
- Module structure finalized in `py-deepbiop/src/pytorch/` (data-model.md, research.md Decision 5)
- All entities delegate to existing crates (data-model.md Section 1-8)
- Zero new external dependencies beyond PyO3/NumPy (api-contracts.md)
- Independently testable via `py-deepbiop/tests/test_pytorch_api.py`

### ✅ II. Multi-Interface Consistency - CONFIRMED PASS

**Design Validation**:
- Python-only API layer as specified (api-contracts.md)
- Core logic shared via delegation pattern (data-model.md validation rules)
- Rust library and CLI remain unchanged (no new interfaces)

### ✅ III. Test-First Development - CONFIRMED PASS

**Design Validation**:
- Comprehensive test plan defined (plan.md lines 85-108)
- Test categories mapped: unit, integration, contract, performance
- All entities have testable contracts (api-contracts.md error handling)
- Examples demonstrate test scenarios (quickstart.md troubleshooting section)

### ✅ IV. Performance as a Feature - CONFIRMED PASS

**Design Validation**:
- GIL release pattern documented (research.md Decision 2, api-contracts.md performance contracts)
- Rayon parallelism preserved via delegation (data-model.md memory flow diagram)
- Streaming I/O and lazy loading specified (data-model.md Dataset entity)
- Performance targets validated in SC-002, SC-005, SC-006 (api-contracts.md)
- Memory flow optimized (data-model.md memory flow diagram)

### ✅ V. Error Handling & User Experience - CONFIRMED PASS

**Design Validation**:
- Error types and messages specified (api-contracts.md error handling section)
- Context provided in all error scenarios (data-model.md validation rules)
- Remediation guidance in quickstart.md (troubleshooting section)
- PyO3 exception conversion documented (research.md Decision 7)

### Final Assessment

**Status**: ✅ ALL CONSTITUTION PRINCIPLES SATISFIED POST-DESIGN

No changes to initial assessment. Design artifacts confirm:
1. Code reuse maximized (wrapper/delegation pattern throughout)
2. Performance preserved (GIL release, Rayon parallelism, zero-copy)
3. Test-first workflow documented
4. User-friendly error messages specified
5. PyTorch compatibility validated

**Design Quality Metrics**:
- Entity count: 8 (Dataset, DataLoader, Transform, Encoder, Augmentation, Sample, Batch, Cache)
- API surface: ~25 public methods across all classes
- Code reuse: 100% (zero duplication, all delegation)
- Estimated implementation size: 500-800 LOC (within scope)

