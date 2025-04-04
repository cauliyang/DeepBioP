# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.15](https://github.com/cauliyang/DeepBioP/compare/deepbiop-cli-v0.1.14...deepbiop-cli-v0.1.15) - 2025-01-22

### Added

- Add markdown help option and documentation for deepbiop-cli
- Update output file extensions for FasToOne and FqsToOne commands to .fa.gz and .fq.gz respectively
- Add output option for specifying file paths in ExtractFa and ExtractFq commands
- Update ExtractFa to support optional read selection by path or random count
- Enhance ExtractFq command to support random selection of reads and update Cargo.toml dependencies
- Add fas2one and fqs2one commands for batch file conversion

## [0.1.14](https://github.com/cauliyang/DeepBioP/compare/deepbiop-cli-v0.1.13...deepbiop-cli-v0.1.14) - 2025-01-17

### Added

- Add tests for k-mer generation and compression detection; update BAM to FASTQ file extension handling
- Integrate deepbiop-core library and refactor module imports across crates
- Introduce deepbiop-core library and refactor encoder options across modules
- Add Fastq to Parquet conversion functionality
- Add stream-based record selection for FASTA and FASTQ
- Add logging information for selected reads and records
- Add extractfq and extractfa commands, and related functionality

### Fixed

- Correct documentation for FqToParquet struct argument names

### Other

- Update function names and improve compression handling in BAM to FASTQ conversion
- add testing data
- Update file extensions and logging messages

## [0.1.13](https://github.com/cauliyang/DeepBioP/compare/deepbiop-cli-v0.1.12...deepbiop-cli-v0.1.13) - 2025-01-07

### Added

- Add fa2parquet conversion functionality
- Add FaToParquet struct for converting fa to parquet

### Fixed

- Update versions of itertools, parquet, arrow and clap-verbosity-flag

## [0.1.11](https://github.com/cauliyang/DeepBioP/compare/deepbiop-cli-v0.1.10...deepbiop-cli-v0.1.11) - 2024-08-25

### Added

- Add FaToFq command for fastq to fasta conversion
- Update keywords in Cargo.toml files
- _(io)_ Refactor bam2fq function to handle equal seq and qual lengths
- Add support for writing compressed fastq files
- Add fastq to fasta conversion command
- Add new file fq2fa.rs
- Add BAM to fastq conversion functionality
- Add CLI installation guide

### Other

- Improve file variable names in FqToFa struct
- Remove unnecessary empty line
- Improve function and field comments
- Refactor file_path assignment in bam2fq.rs
- Merge branch 'dev'

## [0.1.10](https://github.com/cauliyang/DeepBioP/compare/deepbiop-cli-v0.1.9...deepbiop-cli-v0.1.10) - 2024-08-20

### Added

- Update MSRV to 1.75.0
- Add new dependencies and CLI module for chimeric count
- Add deepbiop-cli package with dependencies
- Add support for processing BAM format in DeepBiop
- Add initial project files and configurations

### Fixed

- Update env_logger to version 0.11.5
- Update badge URL for deepbiop crate

### Other

- Update MSRV version in README and improve formatting
- Update description in Cargo.toml
- Remove unnecessary lines in Cargo.toml
- Update dependencies in Cargo.toml and set_up_threads in cli.rs
- Remove outdated documentation badge
- Add documentation badge to README.md
- Update badges in README.md to use consistent format
- Update README.md setup instructions
- Update README.md formatting
- Add tests for bam and fq modules
- Initial commit
