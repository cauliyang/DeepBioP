# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.14](https://github.com/cauliyang/DeepBioP/compare/deepbiop-fa-v0.1.13...deepbiop-fa-v0.1.14) - 2025-01-17

### Added

- Introduce deepbiop-core library and refactor encoder options across modules
- Add stream-based record selection for FASTA and FASTQ
- Add extractfq and extractfa commands, and related functionality

### Other

- Update function names and improve compression handling in BAM to FASTQ conversion
- add docs
- Update function calls for selecting records by stream
- Update file extensions and logging messages

## [0.1.13](https://github.com/cauliyang/DeepBioP/compare/deepbiop-fa-v0.1.12...deepbiop-fa-v0.1.13) - 2025-01-07

### Added

- Add fa2parquet conversion functionality
- Add FaToParquet struct for converting fa to parquet

### Other

- Update PyO3 version to 0.23.3

## [0.1.12](https://github.com/cauliyang/DeepBioP/compare/deepbiop-fa-v0.1.11...deepbiop-fa-v0.1.12) - 2025-01-06

### Added

- Add new modules for JSON and Parquet file I/O
- Add new RNA sequences to test data
- Add initial implementation of deepbiop-fa
- Add CLI installation guide
- Update MSRV to 1.75.0
- Add support for processing BAM format in DeepBiop
- Add initial project files and configurations

### Fixed

- Update Minimum Supported Rust Version to 1.82.0
- Update badge URL for deepbiop crate

### Other

- Update Python object extraction to remove unnecessary variable
- Remove unnecessary file cleanup in test case
- Update function names and use in code
- Update deepbiop-fa crate documentation
- Update file paths in test and Cargo.toml
- Update MSRV version in README and improve formatting
- Remove outdated documentation badge
- Add documentation badge to README.md
- Update badges in README.md to use consistent format
- Update README.md setup instructions
- Update README.md formatting
- Add tests for bam and fq modules
- Initial commit
