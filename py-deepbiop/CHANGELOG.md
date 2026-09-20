# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed (breaking)

- `deepbiop.core` is now only the Rust extension module; the pure-Python `Record`
  dataclass moved to `deepbiop.record`. `from deepbiop.core import Record` is replaced
  by `from deepbiop import Record`.
- `deepbiop.pytorch.Dataset` takes `(file_path, *, sequence_type="dna", transform=None)`;
  the previous underscore-prefixed and never-applied `_sequence_type`/`_transform`/
  `_cache_dir`/`_lazy` keywords are gone. `transform` is now applied to every sample.
- `deepbiop.pytorch.DataLoader` lost the unused `num_workers` argument; use
  `torch.utils.data.DataLoader` for multi-process loading. Its `collate_fn` is now applied.
- `KmerEncoder(canonical=True)` folds reverse complements (output size halves for odd `k`),
  and encoding a sequence shorter than `k` raises `ValueError` instead of returning zeros.
- `Deduplicator.remove_all_duplicates()`/`keep_first` were removed: a single streaming pass
  cannot drop every occurrence of a duplicated sequence. `Deduplicator()` keeps the first
  occurrence of each distinct sequence.
- `VcfReader`/`GtfReader` accept gzip/bgzip input and can be queried repeatedly.
- VCF `Variant.info` is a per-key dict (`Variant.get_info_field(key)` for one lookup)
  rather than a single opaque `raw_info` string.
- `save_cache` stores every sample key and `is_cache_valid` also compares source file size.

### Changed (breaking)

- `deepbiop.core` is now only the Rust extension module; the pure-Python `Record`
  dataclass moved to `deepbiop.record`. `from deepbiop.core import Record` is
  replaced by `from deepbiop import Record` (or `from deepbiop.record import Record`).
- `deepbiop.pytorch.Dataset` takes `(file_path, *, sequence_type="dna", transform=None)`;
  the previous underscore-prefixed and never-applied `_sequence_type`/`_transform`/
  `_cache_dir`/`_lazy` keywords are gone. `transform` is now applied to every sample.
- `deepbiop.pytorch.DataLoader` lost the unused `num_workers` argument; use
  `torch.utils.data.DataLoader` for multi-process loading. Its `collate_fn` is now applied.
- `KmerEncoder(canonical=True)` actually folds reverse complements (output size halves
  for odd `k`), and encoding a sequence shorter than `k` raises `ValueError` instead of
  returning an all-zero vector.
- `Deduplicator.remove_all_duplicates()` / `keep_first` were removed: a single streaming
  pass cannot drop every occurrence of a duplicated sequence. Use `Deduplicator()`,
  which keeps the first occurrence of each distinct sequence.
- `VcfReader`/`GtfReader` read gzip/bgzip input and can be queried repeatedly; parsing
  happens once per reader.
- VCF `Variant.info` is a per-key dict (`Variant.get_info_field(key)` for one lookup)
  rather than a single `raw_info` debug string.
- Cache files store every sample key; `save_cache` validates keys and `load_cache`
  reconstructs them. `is_cache_valid` also compares source file size.

### Added

- **PyTorch-Style Python API**: New `deepbiop.pytorch` module for PyTorch-compatible data loading
  - `Dataset` class for lazy loading of FASTQ/FASTA files with familiar PyTorch interface
  - `DataLoader` class with batching, shuffling, and iteration support
  - Transform classes: `OneHotEncoder`, `IntegerEncoder`, `KmerEncoder` for sequence encoding
  - Augmentation transforms: `Compose`, `ReverseComplement`, `Mutator`, `Sampler`
  - `default_collate` function for variable-length sequence batching with padding
  - Full NumPy/PyTorch tensor compatibility with zero-copy conversion
  - Comprehensive test coverage (21 tests) and working examples
  - Documentation and quickstart guide

## [0.1.4](https://github.com/cauliyang/DeepBioP/compare/py-deepbiop-v0.1.3...py-deepbiop-v0.1.4) - 2024-08-05

### Added

- Add test_fq function
- Add ahash, anyhow, derive_builder, lexical, log
- Add deepbiop-utils crate and Python module
- Add support for processing BAM format in DeepBiop
- Add utils module and get_label_region function
- Update project versions and dependencies
- Add Rye configuration with new dev dependencies
- Add Python module with encoding functionalities
- Add new module 'deepbiop' and 'add' function

### Fixed

- Update maturin version requirement to 1.6.0
- Update Cargo.toml for edition 2021 and dependency versions

### Other

- Update version history in CHANGELOG.md
- Update changelog for version 0.1.3
- Update project URLs in pyproject.toml
- Add tests for bam and fq modules
- Update path in test_load_predict in predicts.rs
- Move fq-related Python functions to deepbiop-fq crate
- Remove unnecessary import lines
- Remove unnecessary spaces and comments
- comment out project.urls Homepage in pyproject.toml
- Add reference to rust-toolchain.toml in pyproject.toml
- Update README.md path in pyproject.toml
- Update location of README.md file
- Remove README.md
- Update release-python.yml and pyproject.toml
- Update project versions and dependencies
- Add initial changelog documentation
- add docs template
- Remove unnecessary relative import in Cargo.toml
- add lock file
- add fastq module

## [0.1.3](https://github.com/cauliyang/DeepBioP/releases/tag/py-deepbiop-v0.1.3) - 2024-08-05

### Added

- Add test_fq function
- Add ahash, anyhow, derive_builder, lexical, log
- Add deepbiop-utils crate and Python module
- Add support for processing BAM format in DeepBiop
- Add utils module and get_label_region function
- Update project versions and dependencies
- Add Rye configuration with new dev dependencies
- Add Python module with encoding functionalities
- Add new module 'deepbiop' and 'add' function

### Fixed

- Update maturin version requirement to 1.6.0
- Update Cargo.toml for edition 2021 and dependency versions

### Other

- Update project URLs in pyproject.toml
- Add tests for bam and fq modules
- Update path in test_load_predict in predicts.rs
- Move fq-related Python functions to deepbiop-fq crate
- Remove unnecessary import lines
- Remove unnecessary spaces and comments
- comment out project.urls Homepage in pyproject.toml
- Add reference to rust-toolchain.toml in pyproject.toml
- Update README.md path in pyproject.toml
- Update location of README.md file
- Remove README.md
- Update release-python.yml and pyproject.toml
- Update project versions and dependencies
- Add initial changelog documentation
- add docs template
- Remove unnecessary relative import in Cargo.toml
- add lock file
- add fastq module

## \[Unreleased\]

## [0.1.0](https://github.com/cauliyang/DeepBioP/releases/tag/py-deepbiop-v0.1.0) - 2024-08-03

### Added

- Add Rye configuration with new dev dependencies
- Add Python module with encoding functionalities
- Add new module 'deepbiop' and 'add' function

### Fixed

- Update Cargo.toml for edition 2021 and dependency versions

### Other

- Add initial changelog documentation
- add docs template
- Remove unnecessary relative import in Cargo.toml
- add lock file
- add fastq module

### Added

- Add Rye configuration with new dev dependencies
- Add Python module with encoding functionalities
- Add new module 'deepbiop' and 'add' function

### Fixed

- Update Cargo.toml for edition 2021 and dependency versions

### Other

- add docs template
- Remove unnecessary relative import in Cargo.toml
- add lock file
- add fastq module
