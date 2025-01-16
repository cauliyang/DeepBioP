use ahash::HashMap;
use ndarray::{Array2, Array3};

pub type Element = i32;
pub type Matrix = Array2<Element>;
pub type Tensor = Array3<Element>;
pub type Kmer2IdTable = HashMap<Vec<u8>, Element>;
pub type Id2KmerTable = HashMap<Element, Vec<u8>>;

pub mod option;
pub mod parquet;
pub mod record;
pub mod traits;

pub use option::*;
pub use parquet::*;
pub use record::*;
pub use traits::*;
