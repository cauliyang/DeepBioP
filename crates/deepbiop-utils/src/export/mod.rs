//! Export utilities for biological data to various formats.
//!
//! This module provides utilities for exporting biological sequence data
//! to formats commonly used in machine learning pipelines:
//! - Parquet (columnar storage) - requires `cache` feature
//! - Arrow (in-memory columnar format) - requires `cache` feature
//! - NumPy (.npy files)

#[cfg(feature = "cache")]
pub mod arrow;
pub mod numpy;
#[cfg(feature = "cache")]
pub mod parquet;
#[cfg(feature = "cache")]
pub mod schema;

#[cfg(feature = "cache")]
pub use arrow::*;
pub use numpy::*;
#[cfg(feature = "cache")]
pub use parquet::*;
#[cfg(feature = "cache")]
pub use schema::*;
