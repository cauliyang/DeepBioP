mod json;
mod parquet;

use std::path::Path;

pub use json::*;
pub use parquet::*;
use std::fs::File;
use std::io;

use anyhow::Result;

/// Check if a file is gzip or bgzip compressed by examining its magic numbers.
///
/// This function reads the first few bytes of a file to detect if it's compressed,
/// without relying on file extensions.
///
/// # Arguments
///
/// * `path` - Path to the file to check
///
/// # Returns
///
/// A Result containing a tuple of two booleans (is_gzip, is_bgzip)
pub fn detect_compression<P: AsRef<Path>>(path: P) -> Result<(bool, bool)> {
    let mut file = File::open(path)?;
    let mut buffer = [0; 4];

    // Read first 4 bytes
    io::Read::read_exact(&mut file, &mut buffer)?;
    // Check gzip magic numbers (1f 8b)
    let is_gzip = buffer[0] == 0x1f && buffer[1] == 0x8b;
    // Check bgzip magic numbers (1f 8b 08 04)
    let is_bgzip = is_gzip && buffer[2] == 0x08 && buffer[3] == 0x04;
    Ok((is_gzip, is_bgzip))
}
