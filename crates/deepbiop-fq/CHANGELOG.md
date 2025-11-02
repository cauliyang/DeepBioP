# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.16](https://github.com/cauliyang/DeepBioP/compare/deepbiop-fq-v0.1.15...deepbiop-fq-v0.1.16) - 2025-11-02

### Added

- add dataset module and integrate with Python bindings

### Other

- ⬆️ chore(deps): upgrade PyO3 to 0.27.1 and fix compatibility issues
- ⚡️ perf(cli): implement streaming for fxs2one to reduce memory footprint
- update MSRV to 1.90.0 and add bon dependency
- Feat ([#68](https://github.com/cauliyang/DeepBioP/pull/68))
- update Minimum Supported Rust Version (MSRV) to 1.85.1 in workflow and README
- *(deps)* update dependencies in Cargo.toml and add dataset.rs file
- Feat ([#57](https://github.com/cauliyang/DeepBioP/pull/57))

## [0.1.15](https://github.com/cauliyang/DeepBioP/compare/deepbiop-fq-v0.1.14...deepbiop-fq-v0.1.15) - 2025-01-22

### Added

- Add random selection functionality for FASTQ records and expose it to Python
- Enhance ExtractFq command to support random selection of reads and update Cargo.toml dependencies
- Add functions to combine multiple FASTA and FASTQ files into bgzip-compressed formats
- Add fas2one and fqs2one commands for batch file conversion

### Other

- ♻️ Optimize record selection in select_record_from_fq_by_random function using for loop
- ♻️ Refactor select_record_from_fq_by_random function to use reservoir sampling algorithm
- Update fetch_records method to use fastq reader and improve record handling

## [0.1.14](https://github.com/cauliyang/DeepBioP/compare/deepbiop-fq-v0.1.13...deepbiop-fq-v0.1.14) - 2025-01-17

### Added

- Integrate deepbiop-core library and refactor module imports across crates
- Introduce deepbiop-core library and refactor encoder options across modules
- Add Fastq to Parquet conversion functionality
- Add stream-based record selection for FASTA and FASTQ
- Add extractfq and extractfa commands, and related functionality

### Other

- Update function names and improve compression handling in BAM to FASTQ conversion
- Update function calls for selecting records by stream
- Update file extensions and logging messages

## [0.1.13](https://github.com/cauliyang/DeepBioP/compare/deepbiop-fq-v0.1.12...deepbiop-fq-v0.1.13) - 2025-01-15

### Added

- Add function py_select_record_from_fq

### Other

- Update PyO3 version to 0.23.3

## [0.1.12](https://github.com/cauliyang/DeepBioP/compare/deepbiop-fq-v0.1.11...deepbiop-fq-v0.1.12) - 2025-01-06

### Added

- Update dependencies versions and add new module

## [0.1.11](https://github.com/cauliyang/DeepBioP/compare/deepbiop-fq-v0.1.10...deepbiop-fq-v0.1.11) - 2024-08-25

### Added
- Update keywords in Cargo.toml files
- Add Python bindings for TensorEncoder
- Add CLI installation guide

### Other
- Merge branch 'dev'

## [0.1.10](https://github.com/cauliyang/DeepBioP/compare/deepbiop-fq-v0.1.9...deepbiop-fq-v0.1.10) - 2024-08-20

### Added
- Update MSRV to 1.75.0

### Other
- Update MSRV version in README and improve formatting

## [0.1.9](https://github.com/cauliyang/DeepBioP/compare/deepbiop-fq-v0.1.8...deepbiop-fq-v0.1.9) - 2024-08-20

### Added
- Update pyo3-stub-gen version to "0.5.2"
- Update Python function signatures and add normalization function

### Other
- Add documentation for chimeric and cigar operations
- Remove outdated documentation badge
- Add documentation badge to README.md
- Remove outdated documentation examples

## [0.1.8](https://github.com/cauliyang/DeepBioP/compare/deepbiop-fq-v0.1.7...deepbiop-fq-v0.1.8) - 2024-08-20

### Added
- Add Python stubs for deepbiop.fq module
- Add pyo3-stub-gen and pyo3-stub-gen methods
- *(io)* Add function to select specific records from Fastq

### Other
- Update deepbiop-fq crate to improve code organization and remove unused functions
- Update deepbiop-fq crate to improve code organization and remove unused functions
- Update deepbiop-bam crate to include chimeric event struct and functions
- Add chimeric event struct and functions to deepbiop-bam crate
- Update badges in README.md to use consistent format

## [0.1.6](https://github.com/cauliyang/DeepBioP/compare/deepbiop-fq-v0.1.5...deepbiop-fq-v0.1.6) - 2024-08-08

### Added
- Add 'anyhow' feature to pyo3 dependency
- Add fastq to fasta conversion function

### Other
- Remove unnecessary features from pyo3 dependencies
- Remove duplicate code for adding submodules
- Improve submodule registration method
- Add sub-module registration for deepbiop modules
- Update README.md setup instructions

## \[Unreleased\]

## [0.1.1](https://github.com/cauliyang/DeepBioP/compare/deepbiop-fq-v0.1.0...deepbiop-fq-v0.1.1) - 2024-08-03

### Added

- Add deepbiop as a new package within the workspace

### Fixed

- Remove test.parquet file after test execution
