# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
