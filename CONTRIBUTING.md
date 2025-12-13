# Contributing to DeepBioP

Thank you for your interest in contributing to DeepBioP! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Development Workflow](#development-workflow)
- [Testing](#testing)
- [Code Style](#code-style)
- [Submitting Changes](#submitting-changes)
- [Release Process](#release-process)

## Code of Conduct

This project follows the [Rust Code of Conduct](https://www.rust-lang.org/policies/code-of-conduct). By participating, you are expected to uphold this code.

## Getting Started

### Prerequisites

- **Rust**: 1.90.0 or later (install via [rustup](https://rustup.rs/))
- **Python**: 3.9 or later (for Python bindings)
- **uv**: Python package manager (recommended, install via `pip install uv`)
- **Git**: Version control

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/DeepBioP.git
   cd DeepBioP
   ```
3. Add upstream remote:
   ```bash
   git remote add upstream https://github.com/cauliyang/DeepBioP.git
   ```

## Development Setup

### Rust Development

```bash
# Build all workspace members
cargo build

# Build with specific features
cargo build --features fastq,bam

# Run tests
cargo test --all

# Format code
cargo fmt --all

# Run linter
cargo clippy --all -- -Dclippy::all
```

### Python Development

```bash
cd py-deepbiop

# Install development dependencies
uv sync

# Build Python bindings in development mode
make build

# Run Python tests
uv run pytest

# Run specific tests
uv run pytest tests/test_fq.py -v

# Lint Python code
uvx ruff check --fix --unsafe-fixes
```

### Pre-commit Hooks

We use pre-commit hooks to ensure code quality:

```bash
# Install pre-commit hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

The hooks include:
- `cargo fmt` (Rust formatting)
- `cargo check` (compilation check)
- `cargo clippy` (linting)
- `ruff` (Python linting/formatting)
- `cargo-sort` (sort Cargo.toml dependencies)

## Development Workflow

### Creating a Branch

```bash
# Sync with upstream
git fetch upstream
git checkout main
git merge upstream/main

# Create a feature branch
git checkout -b feature/your-feature-name
```

### Making Changes

1. **Write tests first** (TDD approach preferred)
2. **Implement the feature or fix**
3. **Ensure all tests pass**
4. **Update documentation** as needed

### Commit Messages

Follow the [Conventional Commits](https://www.conventionalcommits.org/) format:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Test additions or modifications
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `chore`: Build/tool changes

Examples:
```
feat(fq): add k-mer encoding with canonical support

Implements k-mer frequency encoding with optional canonical k-mer
mode (treating a k-mer and its reverse complement as the same).

Closes #42
```

```
fix(gtf): handle GTF-specific attribute format

The noodles GFF library expects GFF3 format but GTF uses different
attribute syntax. Implemented custom parser for GTF attributes.
```

## Testing

### Rust Tests

```bash
# Run all tests
cargo test --workspace

# Run tests for specific crate
cargo test -p deepbiop-fq

# Run tests with output
cargo test -- --nocapture

# Run doctests
cargo test --doc
```

### Python Tests

```bash
cd py-deepbiop

# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=deepbiop --cov-report=html

# Run specific test class
uv run pytest tests/test_fq_encoding.py::TestOneHotEncoder -v
```

### Adding Tests

- **Rust**: Add unit tests in `#[cfg(test)]` modules within source files
- **Python**: Add tests in `py-deepbiop/tests/` following existing patterns
- **Test data**: Place in `crates/*/tests/data/` or `py-deepbiop/tests/data/`

## Code Style

### Rust

- Follow the [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)
- Use `cargo fmt` for automatic formatting
- Address all `cargo clippy` warnings
- Add documentation comments (`///`) for public items
- Include examples in documentation

### Python

- Follow [PEP 8](https://pep8.org/)
- Use `ruff` for linting and formatting
- Add type hints for all function signatures
- Write docstrings in NumPy style

### Documentation

- Update README.md for user-facing changes
- Update CHANGELOG.md following [Keep a Changelog](https://keepachangelog.com/)
- Add rustdoc comments for Rust APIs
- Update Python docstrings for Python APIs
- Include code examples where appropriate

## Submitting Changes

### Pull Request Process

1. **Ensure all tests pass**:
   ```bash
   cargo test --all
   cd py-deepbiop && uv run pytest
   ```

2. **Run code quality checks**:
   ```bash
   cargo fmt --all
   cargo clippy --all -- -Dclippy::all
   cd py-deepbiop && uvx ruff check --fix --unsafe-fixes
   ```

3. **Update documentation**:
   - Update CHANGELOG.md with your changes
   - Update relevant documentation files
   - Regenerate API docs if needed

4. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

5. **Create Pull Request**:
   - Go to GitHub and create a PR from your branch
   - Fill in the PR template with details
   - Link any related issues
   - Request review from maintainers

### PR Requirements

- ✅ All CI checks pass (tests, linting, formatting)
- ✅ Code coverage doesn't decrease
- ✅ Documentation updated
- ✅ CHANGELOG.md updated
- ✅ Commit messages follow conventional commits
- ✅ PR description explains changes and rationale

## Project Structure

```
DeepBioP/
├── crates/                 # Rust crates
│   ├── deepbiop/          # Umbrella crate
│   ├── deepbiop-core/     # Core types and utilities
│   ├── deepbiop-fq/       # FASTQ processing
│   ├── deepbiop-fa/       # FASTA processing
│   ├── deepbiop-bam/      # BAM processing
│   ├── deepbiop-gtf/      # GTF processing
│   ├── deepbiop-vcf/      # VCF processing
│   ├── deepbiop-utils/    # Shared utilities
│   └── deepbiop-cli/      # CLI tool
├── py-deepbiop/           # Python bindings
│   ├── src/              # Rust code for Python module
│   └── tests/            # Python tests
├── docs/                  # Documentation
└── specs/                 # Feature specifications
```

## Architecture Guidelines

### Rust Crates

- **Core crate** (`deepbiop-core`): Shared types, traits, and utilities
- **Format crates** (`deepbiop-{fq,fa,bam,gtf,vcf}`): Format-specific I/O and processing
- **Utils crate** (`deepbiop-utils`): Export, I/O helpers, common operations
- **CLI crate** (`deepbiop-cli`): Command-line interface
- **Umbrella crate** (`deepbiop`): Re-exports with feature flags

### Python Bindings

- Each Rust crate has optional `python` feature for PyO3 bindings
- `py-deepbiop` aggregates all Python modules
- Use `#[pyclass]` for exposed types
- Use `#[pymethods]` for exposed methods
- Provide both item-level and batch methods

### Adding a New Feature

1. **Rust implementation**:
   - Implement in appropriate crate
   - Add comprehensive tests
   - Add documentation with examples

2. **Python bindings** (if applicable):
   - Enable `python` feature in crate's `Cargo.toml`
   - Create `python.rs` module with PyO3 bindings
   - Update `py-deepbiop/src/lib.rs` to register module
   - Add Python tests

3. **CLI command** (if applicable):
   - Add struct in `crates/deepbiop-cli/src/cli/`
   - Add to `Commands` enum
   - Implement logic using library crates

## Release Process

(Maintainers only)

1. Update version in all `Cargo.toml` files
2. Update CHANGELOG.md
3. Create git tag: `git tag -a v0.2.0 -m "Release v0.2.0"`
4. Push tag: `git push origin v0.2.0`
5. CI will automatically publish to crates.io and PyPI

## Getting Help

- **Questions**: Open a [discussion](https://github.com/cauliyang/DeepBioP/discussions)
- **Bugs**: Open an [issue](https://github.com/cauliyang/DeepBioP/issues)
- **Feature requests**: Open an issue with the `enhancement` label
- **Security issues**: Email maintainers directly (see SECURITY.md)

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (MIT/Apache-2.0).

## Recognition

Contributors will be recognized in:
- CHANGELOG.md for their specific contributions
- GitHub contributors page
- Release notes

Thank you for contributing to DeepBioP!
