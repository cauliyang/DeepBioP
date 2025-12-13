//! PyTorch-style Python API for biological sequence data loading.
//!
//! This module provides PyTorch-compatible Dataset and DataLoader classes
//! for loading and preprocessing FASTQ/FASTA files, enabling researchers
//! to use familiar PyTorch patterns with biological sequence data.
//!
//! # Example
//!
//! ```python
//! from deepbiop.pytorch import Dataset, DataLoader, OneHotEncoder
//!
//! # Create dataset with encoding
//! dataset = Dataset("data.fastq", transform=OneHotEncoder())
//!
//! # Create data loader
//! loader = DataLoader(dataset, batch_size=32, shuffle=True)
//!
//! # Iterate through batches
//! for batch in loader:
//!     sequences = batch['sequences']  # NumPy array
//!     # ... training logic
//! ```

use pyo3::prelude::*;

// Module declarations (files created in subsequent tasks)
pub mod cache;
pub mod collate; // T030
pub mod dataloader; // T025-T027
pub mod dataset; // T012-T016
pub mod errors;
pub mod transforms; // T018-T022, T036-T042
pub mod types; // T053-T056

/// Register the pytorch module with Python.
///
/// This function is called from lib.rs to expose the pytorch submodule
/// to Python as `deepbiop.pytorch`.
pub fn register_pytorch_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let pytorch_module = PyModule::new(parent_module.py(), "pytorch")?;

    // Module docstring
    pytorch_module.add(
        "__doc__",
        "PyTorch-style Python API for biological sequence data loading.\n\n\
         Provides Dataset, DataLoader, and transform classes for loading FASTQ/FASTA files\n\
         with familiar PyTorch patterns.",
    )?;

    // Register submodule classes (will be added as implementation progresses)
    pytorch_module.add_class::<dataset::Dataset>()?;
    pytorch_module.add_class::<dataset::DatasetIterator>()?;

    // Register dataloader classes
    pytorch_module.add_class::<dataloader::DataLoader>()?;
    pytorch_module.add_class::<dataloader::DataLoaderIterator>()?;

    // Register transform classes
    pytorch_module.add_class::<transforms::OneHotEncoder>()?;
    pytorch_module.add_class::<transforms::IntegerEncoder>()?;
    pytorch_module.add_class::<transforms::KmerEncoder>()?;
    pytorch_module.add_class::<transforms::Compose>()?;
    pytorch_module.add_class::<transforms::ReverseComplement>()?;
    pytorch_module.add_class::<transforms::Mutator>()?;
    pytorch_module.add_class::<transforms::Sampler>()?;

    // Register collate functions
    pytorch_module.add_function(wrap_pyfunction!(collate::default_collate, &pytorch_module)?)?;

    // Register cache functions
    cache::register_cache_functions(&pytorch_module)?;

    // Add pytorch submodule to parent
    parent_module.add_submodule(&pytorch_module)?;

    Ok(())
}
