# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.11](https://github.com/cauliyang/DeepBioP/compare/deepbiop-bam-v0.1.10...deepbiop-bam-v0.1.11) - 2024-08-25

### Added
- Update keywords in Cargo.toml files
- *(io)* Refactor bam2fq function to handle equal seq and qual lengths
- Add BAM to fastq conversion functionality
- Add CLI installation guide

### Other
- Remove unnecessary empty line
- Improve function and field comments
- Improve code readability and performance
- Merge branch 'dev'

## [0.1.10](https://github.com/cauliyang/DeepBioP/compare/deepbiop-bam-v0.1.9...deepbiop-bam-v0.1.10) - 2024-08-20

### Added
- Update MSRV to 1.75.0

### Other
- Update MSRV version in README and improve formatting

## [0.1.9](https://github.com/cauliyang/DeepBioP/compare/deepbiop-bam-v0.1.8...deepbiop-bam-v0.1.9) - 2024-08-20

### Added
- Add functions to count chimeric reads in BAM files

### Other
- Add documentation for chimeric and cigar operations
- Remove outdated documentation badge
- Add documentation badge to README.md

## [0.1.8](https://github.com/cauliyang/DeepBioP/compare/deepbiop-bam-v0.1.7...deepbiop-bam-v0.1.8) - 2024-08-20

### Added
- Add pyo3-stub-gen dependency and stub generation

### Other
- Add example for parsing sa tag string
- Update deepbiop-fq crate to improve code organization and remove unused functions
- Update deepbiop-bam crate to include chimeric event struct and functions
- Add chimeric event struct and functions to deepbiop-bam crate
- Add function documentation for chimeric events
- Update badges in README.md to use consistent format

## [0.1.7](https://github.com/cauliyang/DeepBioP/compare/deepbiop-bam-v0.1.6...deepbiop-bam-v0.1.7) - 2024-08-11

### Added
- Add new chimeric event example in documentation
- Add debug log for parsing SA tag
- Add chimeric read filtering and checking functions
- Add derive_builder and deepbiop-utils dependencies

### Fixed
- Update import statement for logging module

### Other
- Change name field in ChimericEvent to be an Option
- Rename functions for clarity and consistency
- Rename function filter_reads to keep_reads

## [0.1.6](https://github.com/cauliyang/DeepBioP/compare/deepbiop-bam-v0.1.5...deepbiop-bam-v0.1.6) - 2024-08-08

### Added
- Add Segment class with overlap method
- Add chimeric event module
- Add rayon and ahash dependencies, implement chimeric.rs

### Other
- Remove duplicate code for adding submodules
- Improve submodule registration method
- Add sub-module registration for deepbiop modules
- Improve chimeric read processing functions
- Update README.md setup instructions
