//! **deepbiop-vcf** provides VCF (Variant Call Format) file processing for deep learning preprocessing.
//!
//! This crate enables reading, filtering, and extracting features from genomic variant files.
//!
//! # Example
//!
//! ```no_run
//! use deepbiop_vcf::reader::VcfReader;
//! use deepbiop_vcf::filter::VariantFilter;
//! use std::path::Path;
//!
//! // Read VCF file
//! let mut reader = VcfReader::open(Path::new("variants.vcf")).unwrap();
//! let variants = reader.read_all().unwrap();
//!
//! // Filter high-quality SNPs
//! let filter = VariantFilter::new()
//!     .with_min_quality(30.0)
//!     .with_variant_types(vec!["SNP".to_string()])
//!     .pass_only();
//!
//! let filtered = filter.apply(&variants);
//! println!("Found {} high-quality SNPs", filtered.len());
//! ```

pub mod annotate;
pub mod filter;
pub mod reader;
pub mod types;

#[cfg(feature = "python")]
pub mod python;

pub use annotate::InfoExtractor;
pub use filter::VariantFilter;
pub use reader::VcfReader;
pub use types::Variant;

#[cfg(feature = "python")]
pub use python::register_vcf_module;
