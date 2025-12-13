use ahash::HashMap;
use ndarray::{Array2, Array3};

pub mod augmentation;
pub mod encoding;

pub type Element = i32;
pub type Matrix = Array2<Element>;
pub type Tensor = Array3<Element>;
pub type Kmer2IdTable = HashMap<Vec<u8>, Element>;
pub type Id2KmerTable = HashMap<Element, Vec<u8>>;

// Re-export encoding types for convenience
pub use encoding::{EncodingScheme, EncodingType};

// Re-export augmentation types for convenience
pub use augmentation::{AugmentationTransform, TransformType};
