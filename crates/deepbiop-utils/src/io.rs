mod json;
#[cfg(feature = "cache")]
mod parquet;

use std::path::Path;

use anyhow::Result;
pub use json::*;
#[cfg(feature = "cache")]
pub use parquet::*;
use std::fs::File;

use flate2::read::GzDecoder;
use noodles::bgzf;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3_stub_gen::derive::*;
use std::io;
use std::io::Read;
use std::path::PathBuf;

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
#[cfg_attr(feature = "python", gen_stub_pyclass_enum)]
#[cfg_attr(feature = "python", pyclass(eq, eq_int, module = "deepbiop.utils"))]
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
pub fn create_reader_for_compressed_file<P: AsRef<Path>>(
    file_path: P,
) -> Result<Box<dyn io::Read>> {
    let compressed_type = check_compressed_type(file_path.as_ref())?;
    let file = File::open(file_path)?;

    Ok(match compressed_type {
        CompressedType::Uncompress => Box::new(file),
        CompressedType::Gzip => Box::new(GzDecoder::new(file)),
        CompressedType::Bgzip => Box::new(bgzf::io::Reader::new(file)),
        _ => return Err(anyhow::anyhow!("unsupported compression type")),
    })
}

/// Creates a multithreaded BGZip reader for compressed files
///
/// This function creates a reader optimized for reading BGZip-compressed files
/// using multiple threads for decompression. The thread count is automatically
/// capped at the system's available parallelism.
///
/// # Arguments
///
/// * `file_path` - Path to the BGZip-compressed file
/// * `threads` - Optional number of threads to use (defaults to 2 if None)
///
/// # Returns
///
/// * `Ok(bgzf::io::Reader<File>)` - A multithreaded BGZip reader
/// * `Err` - If the file cannot be opened
///
/// # Examples
///
/// ```no_run
/// use deepbiop_utils::io::create_multithreaded_reader;
/// use std::path::Path;
///
/// let reader = create_multithreaded_reader(Path::new("data.fq.gz"), Some(4)).unwrap();
/// // Use reader with noodles parsers
/// ```
pub fn create_multithreaded_reader<P: AsRef<Path>>(
    file_path: P,
    threads: Option<usize>,
) -> Result<bgzf::io::Reader<File>> {
    use crate::parallel::calculate_worker_count;

    let worker_count = calculate_worker_count(threads);
    let file = File::open(file_path)?;

    // Note: noodles bgzf::io::Reader doesn't currently expose worker count
    // configuration in the same way as Writer. This creates a standard reader
    // that will use internal threading. For now, we return a standard reader.
    // The worker_count calculation is kept for API consistency and future use.
    let _ = worker_count; // Acknowledge unused variable for now

    Ok(bgzf::io::Reader::new(file))
}

/// Creates a multithreaded BGZip writer for compressed files
///
/// This function creates a writer optimized for writing BGZip-compressed files
/// using multiple threads for compression. The thread count is automatically
/// capped at the system's available parallelism.
///
/// # Arguments
///
/// * `file_path` - Path where the BGZip-compressed file will be written
/// * `threads` - Optional number of threads to use (defaults to 2 if None)
///
/// # Returns
///
/// * `Ok(bgzf::io::MultithreadedWriter<File>)` - A multithreaded BGZip writer
/// * `Err` - If the file cannot be created
///
/// # Examples
///
/// ```no_run
/// use deepbiop_utils::io::create_multithreaded_writer;
/// use std::path::Path;
///
/// let writer = create_multithreaded_writer(Path::new("output.fq.gz"), Some(4)).unwrap();
/// // Use writer with noodles writers
/// ```
pub fn create_multithreaded_writer<P: AsRef<Path>>(
    file_path: P,
    threads: Option<usize>,
) -> Result<bgzf::io::MultithreadedWriter<File>> {
    use crate::parallel::calculate_worker_count;

    let worker_count = calculate_worker_count(threads.or(Some(2)));
    let file = File::create(file_path)?;

    Ok(bgzf::io::MultithreadedWriter::with_worker_count(
        worker_count,
        file,
    ))
}

/// Streams and merges records from multiple files into a single output file
///
/// This is a generic function that processes multiple input files sequentially,
/// reading records one at a time and writing them to a single output file without
/// loading all data into memory. This is memory-efficient for large biological datasets.
///
/// # Type Parameters
///
/// * `P` - Path type that can be converted to `AsRef<Path>`
/// * `R` - Reader type that implements `Iterator<Item = Result<T, E>>`
/// * `W` - Writer type that can write records
/// * `T` - Record type being processed
/// * `E` - Error type from the reader
/// * `CreateReader` - Function that creates a reader from a path
/// * `CreateWriter` - Function that creates a writer from a path and thread count
/// * `WriteRecord` - Function that writes a single record
///
/// # Arguments
///
/// * `paths` - Slice of input file paths to merge
/// * `result_path` - Path where the merged output will be written
/// * `threads` - Optional number of threads for the writer (if supported)
/// * `create_reader` - Closure that creates a reader for a given path
/// * `create_writer` - Closure that creates a writer for the output path
/// * `write_record` - Closure that writes a single record to the writer
///
/// # Returns
///
/// * `Ok(())` - If all files were successfully merged
/// * `Err` - If any file operation fails
///
/// # Examples
///
/// Basic usage pattern (implementation details omitted for brevity):
///
/// ```rust,ignore
/// use deepbiop_utils::io::stream_merge_records;
/// use std::path::PathBuf;
/// use anyhow::Result;
///
/// // This function demonstrates the pattern for merging FASTQ files
/// // The actual implementation would need to handle reader/writer lifetimes appropriately
/// fn merge_fastq_files(paths: &[PathBuf], output: PathBuf) -> Result<()> {
///     stream_merge_records(
///         paths,
///         output.as_path(),
///         Some(2),
///         create_fastq_reader,  // Function that creates a reader iterator
///         create_fastq_writer,  // Function that creates a writer
///         write_fastq_record,   // Function that writes a single record
///     )
/// }
/// ```
pub fn stream_merge_records<P, R, W, T, E, CreateReader, CreateWriter, WriteRecord>(
    paths: &[PathBuf],
    result_path: P,
    threads: Option<usize>,
    create_reader: CreateReader,
    create_writer: CreateWriter,
    write_record: WriteRecord,
) -> Result<()>
where
    P: AsRef<Path>,
    R: Iterator<Item = std::result::Result<T, E>>,
    E: Into<anyhow::Error>,
    CreateReader: Fn(&Path) -> Result<R>,
    CreateWriter: Fn(&Path, Option<usize>) -> Result<W>,
    WriteRecord: Fn(&mut W, &T) -> Result<()>,
{
    log::info!(
        "Merging {} files to {:?}",
        paths.len(),
        result_path.as_ref()
    );

    let mut writer = create_writer(result_path.as_ref(), threads)?;

    // Process each file sequentially in a streaming fashion
    for path in paths {
        log::info!("Processing file: {:?}", path);
        let reader = create_reader(path)?;

        // Stream records one at a time without loading all into memory
        for result in reader {
            let record = result.map_err(|e| e.into())?;
            write_record(&mut writer, &record)?;
        }
    }

    log::info!("Successfully merged {} files", paths.len());
    Ok(())
}

/// Represents different types of sequence file formats
#[cfg_attr(feature = "python", gen_stub_pyclass_enum)]
#[cfg_attr(feature = "python", pyclass(eq, eq_int, module = "deepbiop.utils"))]
#[derive(Debug, PartialEq, Eq, Clone, Hash)]
pub enum SequenceFileType {
    Fasta,
    Fastq,
    Unknown,
}

/// Determines if a file is FASTA or FASTQ format by checking its first character
///
/// # Arguments
/// * `file_path` - Path to the sequence file (can be compressed or uncompressed)
///
/// # Returns
/// * `Ok(SequenceFileType)` - The detected sequence file type
/// * `Err` - If there was an error reading the file
///
/// # Example
/// ```no_run
/// use deepbiop_utils::io;
///
/// let file_type = io::check_sequence_file_type("sample.fq").unwrap();
/// ```
pub fn check_sequence_file_type<P: AsRef<Path>>(file_path: P) -> Result<SequenceFileType> {
    let mut reader = create_reader_for_compressed_file(file_path)?;
    let mut buffer = [0u8; 1];

    // Read the first byte
    match reader.read_exact(&mut buffer) {
        Ok(_) => match buffer[0] as char {
            '>' => Ok(SequenceFileType::Fasta),
            '@' => Ok(SequenceFileType::Fastq),
            _ => Ok(SequenceFileType::Unknown),
        },
        Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => Ok(SequenceFileType::Unknown),
        Err(e) => Err(e.into()),
    }
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

    #[test]
    fn test_sequence_file_type() -> Result<()> {
        let test_fq = "./tests/data/test.fastq";
        assert_eq!(check_sequence_file_type(test_fq)?, SequenceFileType::Fastq);

        let test_fa = "./tests/data/test.fa.gz";
        assert_eq!(check_sequence_file_type(test_fa)?, SequenceFileType::Fasta);

        let test_compresed_fq = "./tests/data/test.fastq.gz";
        assert_eq!(
            check_sequence_file_type(test_compresed_fq)?,
            SequenceFileType::Fastq
        );
        Ok(())
    }
}
