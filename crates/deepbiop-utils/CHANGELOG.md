# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.11](https://github.com/cauliyang/DeepBioP/compare/deepbiop-utils-v0.1.10...deepbiop-utils-v0.1.11) - 2024-08-25

### Added
- Update keywords in Cargo.toml files
- Add CLI installation guide

### Other
- Merge branch 'dev'

## [0.1.10](https://github.com/cauliyang/DeepBioP/compare/deepbiop-utils-v0.1.9...deepbiop-utils-v0.1.10) - 2024-08-20

### Added
- Update MSRV to 1.75.0

### Other
- Update MSRV version in README and improve formatting

## [0.1.9](https://github.com/cauliyang/DeepBioP/compare/deepbiop-utils-v0.1.8...deepbiop-utils-v0.1.9) - 2024-08-20

### Added
- Update function definitions and imports

### Other
- Add documentation for chimeric and cigar operations
- Remove outdated documentation badge
- Add documentation badge to README.md

## [0.1.8](https://github.com/cauliyang/DeepBioP/compare/deepbiop-utils-v0.1.7...deepbiop-utils-v0.1.8) - 2024-08-20

### Added
- Add pyo3-stub-gen dependency and stub generation

### Other
- Update deepbiop-fq crate to improve code organization and remove unused functions
- Update deepbiop-bam crate to include chimeric event struct and functions
- Update badges in README.md to use consistent format

## [0.1.7](https://github.com/cauliyang/DeepBioP/compare/deepbiop-utils-v0.1.6...deepbiop-utils-v0.1.7) - 2024-08-11

### Added
- Add derive_builder and deepbiop-utils dependencies

## [0.1.6](https://github.com/cauliyang/DeepBioP/compare/deepbiop-utils-v0.1.5...deepbiop-utils-v0.1.6) - 2024-08-08

### Added
- Add validation to GenomicIntervalBuilder
- Add GenomicInterval2 class and test in utils module
- Add Python bindings for Segment struct
- Add deepbiop.utils submodule and Segment import
- Add Segment class with overlap method
- Add chimeric event module

### Fixed
- Fix typo in variable name in Python module registration
- Correct class import in python module

### Other
- Update visibility of methods in GenomicInterval
- Remove commented-out code and unnecessary imports
- Remove unused code and reorganize structure
- improve code readability and remove comments
- Change visibility of methods in GenomicInterval and GenomicInterval2
- Rename Segment to GenomicInterval and move to genomics
- Remove unused import and class declaration
- Remove unused SegmentBuilder code
- Remove duplicate code for adding submodules
- Improve submodule registration method
- Use variable directly in function call for module name
- Improve chimeric read processing functions
- Update README.md setup instructions
