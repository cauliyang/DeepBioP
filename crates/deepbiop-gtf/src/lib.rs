//! **deepbiop-gtf** provides GTF (Gene Transfer Format) annotation file processing.
//!
//! This crate enables reading, querying, and analyzing genomic annotations for deep learning preprocessing.
//!
//! # Example
//!
//! ```no_run
//! use deepbiop_gtf::reader::GtfReader;
//! use deepbiop_gtf::query::{GeneQuery, RegionQuery};
//! use std::path::Path;
//!
//! // Read GTF file
//! let mut reader = GtfReader::open(Path::new("annotations.gtf")).unwrap();
//! let features = reader.read_all().unwrap();
//!
//! // Query by genomic region
//! let region_features = RegionQuery::query(&features, "chr1", 1000000, 2000000);
//! println!("Found {} features in region", region_features.len());
//!
//! // Build gene index for fast lookups
//! let mut reader2 = GtfReader::open(Path::new("annotations.gtf")).unwrap();
//! let gene_index = reader2.build_gene_index().unwrap();
//! let gene_query = GeneQuery::new(gene_index);
//!
//! if let Some(gene_features) = gene_query.get_gene("ENSG00000000001") {
//!     println!("Gene has {} features", gene_features.len());
//! }
//! ```

pub mod query;
pub mod reader;
pub mod types;

#[cfg(feature = "python")]
pub mod python;

pub use query::{GeneQuery, RegionQuery};
pub use reader::GtfReader;
pub use types::{GenomicFeature, Strand};

#[cfg(feature = "python")]
pub use python::register_gtf_module;
