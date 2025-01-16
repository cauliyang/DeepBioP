mod json;
mod parquet;

use std::path::Path;

use anyhow::Result;
pub use json::*;
pub use parquet::*;
use std::fs::File;

use flate2::read::GzDecoder;
use noodles::bgzf;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::*;
use std::io;
use std::io::Read;

/// Represents different types of file compression formats
///
/// This enum is used to identify and handle various compression formats commonly used for files.
/// It can be used in Python through the deepbiop.utils module.
///
/// # Variants
///
/// * `Uncompress` - Uncompressed/raw file format
/// * `Gzip` - Standard gzip compression (.gz files)
/// * `Bgzip` - Blocked gzip format, commonly used in bioinformatics
/// * `Zip` - ZIP archive format
/// * `Bzip2` - bzip2 compression format
/// * `Xz` - XZ compression format (LZMA2)
/// * `Zstd` - Zstandard compression format
/// * `Unknown` - Unknown or unrecognized compression format
#[gen_stub_pyclass_enum]
#[pyclass(eq, eq_int, module = "deepbiop.utils")]
#[derive(Debug, PartialEq, Clone, Eq, Hash)]
pub enum CompressedType {
    Uncompress,
    Gzip,
    Bgzip,
    Zip,
    Bzip2,
    Xz,
    Zstd,
    Unknown,
}

/// Determines the compression type of a file by examining its header/signature
///
/// This function reads the first few bytes of a file and checks for known magic numbers
/// or file signatures to identify the compression format used.
///
/// # Arguments
///
/// * `file_path` - Path to the file to check, can be any type that converts to a Path
///
/// # Returns
///
/// * `Result<CompressedType>` - The detected compression type wrapped in a Result
///
/// # Errors
///
/// Returns an error if:
/// * The file cannot be opened
/// * There are issues reading the file header
///
/// # Examples
///
/// ```no_run
/// use deepbiop_utils::io::check_compressed_type;
/// use std::path::Path;
///
/// let file_path = Path::new("test.gz");
/// let compression = check_compressed_type(file_path).unwrap();
/// ```
pub fn check_compressed_type<P: AsRef<Path>>(file_path: P) -> Result<CompressedType> {
    let mut file = File::open(file_path)?;
    let mut buffer = [0u8; 18]; // Large enough for BGZF detection

    // Read the first few bytes
    let bytes_read = file.read(&mut buffer)?;
    if bytes_read < 2 {
        return Ok(CompressedType::Uncompress);
    }

    // Check magic numbers/file signatures
    match &buffer[..] {
        // Check for BGZF first (starts with gzip magic number + specific extra fields)
        [0x1f, 0x8b, 0x08, 0x04, ..] if bytes_read >= 18 => {
            // Check for BGZF extra field
            let xlen = u16::from_le_bytes([buffer[10], buffer[11]]) as usize;
            if xlen >= 6 && buffer[12] == 0x42  // B
                && buffer[13] == 0x43  // C
                && buffer[14] == 0x02  // Length of subfield (2)
                && buffer[15] == 0x00
            // Length of subfield (2)
            {
                Ok(CompressedType::Bgzip)
            } else {
                Ok(CompressedType::Gzip)
            }
        }

        // Regular Gzip: starts with 0x1F 0x8B
        [0x1f, 0x8b, ..] => Ok(CompressedType::Gzip),

        // Zip: starts with "PK\x03\x04" or "PK\x05\x06" (empty archive) or "PK\x07\x08" (spanned archive)
        [0x50, 0x4b, 0x03, 0x04, ..]
        | [0x50, 0x4b, 0x05, 0x06, ..]
        | [0x50, 0x4b, 0x07, 0x08, ..] => Ok(CompressedType::Zip),

        // Bzip2: starts with "BZh"
        [0x42, 0x5a, 0x68, ..] => Ok(CompressedType::Bzip2),

        // XZ: starts with 0xFD "7zXZ"
        [0xfd, 0x37, 0x7a, 0x58, 0x5a, 0x00, ..] => Ok(CompressedType::Xz),

        // Zstandard: starts with magic number 0xFD2FB528
        [0x28, 0xb5, 0x2f, 0xfd, ..] => Ok(CompressedType::Zstd),

        // If no compression signature is found, assume it's a normal file
        _ => {
            // Additional check for text/binary content could be added here
            Ok(CompressedType::Uncompress)
        }
    }
}

/// Checks if a file is compressed by examining its file signature/magic numbers
///
/// # Arguments
/// * `file_path` - Path to the file to check
///
/// # Returns
/// * `Ok(true)` if the file is compressed (gzip, bgzip, zip, bzip2, xz, zstd)
/// * `Ok(false)` if the file is uncompressed or compression type is unknown
/// * `Err` if there was an error reading the file
///
/// # Example
/// ```no_run
/// use deepbiop_utils::io;
///
/// let is_compressed = io::is_compressed("file.gz").unwrap();
/// assert!(is_compressed);
/// ```
pub fn is_compressed<P: AsRef<Path>>(file_path: P) -> Result<bool> {
    match check_compressed_type(file_path)? {
        CompressedType::Uncompress => Ok(false),
        CompressedType::Unknown => Ok(false),
        _ => Ok(true),
    }
}

/// Creates a reader for a file that may be compressed
///
/// This function detects the compression type of the file and returns an appropriate reader.
/// Currently supports uncompressed files, gzip, and bgzip formats.
///
/// # Arguments
/// * `file_path` - Path to the file to read, can be compressed or uncompressed
///
/// # Returns
/// * `Ok(Box<dyn io::Read>)` - A boxed reader appropriate for the file's compression
/// * `Err` - If the file cannot be opened or has an unsupported compression type
pub fn create_reader<P: AsRef<Path>>(file_path: P) -> Result<Box<dyn io::Read>> {
    let compressed_type = check_compressed_type(file_path.as_ref())?;
    let file = File::open(file_path)?;

    Ok(match compressed_type {
        CompressedType::Uncompress => Box::new(file),
        CompressedType::Gzip => Box::new(GzDecoder::new(file)),
        CompressedType::Bgzip => Box::new(bgzf::Reader::new(file)),
        _ => return Err(anyhow::anyhow!("unsupported compression type")),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_check_file_type() -> Result<()> {
        // Test gzip file
        let mut gzip_file = NamedTempFile::new()?;
        gzip_file.write_all(&[0x1f, 0x8b])?;
        assert_eq!(
            check_compressed_type(gzip_file.path())?,
            CompressedType::Gzip
        );

        // Test bgzip file
        let mut bgzip_file = NamedTempFile::new()?;
        let bgzip_header = [
            0x1f, 0x8b, 0x08, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x06, 0x00, 0x42, 0x43,
            0x02, 0x00, 0x00, 0x00,
        ];
        bgzip_file.write_all(&bgzip_header)?;
        assert_eq!(
            check_compressed_type(bgzip_file.path())?,
            CompressedType::Bgzip
        );

        // Test zip file
        let mut zip_file = NamedTempFile::new()?;
        zip_file.write_all(&[0x50, 0x4b, 0x03, 0x04])?;
        assert_eq!(check_compressed_type(zip_file.path())?, CompressedType::Zip);

        // Test bzip2 file
        let mut bzip2_file = NamedTempFile::new()?;
        bzip2_file.write_all(&[0x42, 0x5a, 0x68])?;
        assert_eq!(
            check_compressed_type(bzip2_file.path())?,
            CompressedType::Bzip2
        );

        // Test xz file
        let mut xz_file = NamedTempFile::new()?;
        xz_file.write_all(&[0xfd, 0x37, 0x7a, 0x58, 0x5a, 0x00])?;
        assert_eq!(check_compressed_type(xz_file.path())?, CompressedType::Xz);

        // Test zstd file
        let mut zstd_file = NamedTempFile::new()?;
        zstd_file.write_all(&[0x28, 0xb5, 0x2f, 0xfd])?;
        assert_eq!(
            check_compressed_type(zstd_file.path())?,
            CompressedType::Zstd
        );

        // Test normal file
        let mut normal_file = NamedTempFile::new()?;
        normal_file.write_all(b"Hello world")?;
        assert_eq!(
            check_compressed_type(normal_file.path())?,
            CompressedType::Uncompress
        );

        Ok(())
    }

    #[test]
    fn test_is_compressed() -> Result<()> {
        // Test compressed file
        let mut gzip_file = NamedTempFile::new()?;
        gzip_file.write_all(&[0x1f, 0x8b])?;
        assert!(is_compressed(gzip_file.path())?);

        // Test uncompressed file
        let mut normal_file = NamedTempFile::new()?;
        normal_file.write_all(b"Hello world")?;
        assert!(!is_compressed(normal_file.path())?);

        Ok(())
    }

    #[test]
    fn test_real_example() -> Result<()> {
        let test1 = "./tests/data/test.fastq.gz";
        let test2 = "./tests/data/test.fastqbgz.gz";
        let test3 = "./tests/data/test.fastq";

        assert_eq!(check_compressed_type(test1)?, CompressedType::Gzip);
        assert_eq!(check_compressed_type(test2)?, CompressedType::Bgzip);
        assert_eq!(check_compressed_type(test3)?, CompressedType::Uncompress);
        Ok(())
    }
}
