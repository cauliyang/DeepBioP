use derive_builder::Builder;
use serde::{Deserialize, Serialize};
use std::fmt::{self, Display, Formatter};

use pyo3::prelude::*;
use pyo3_stub_gen::derive::*;

pub const BASES: &[u8] = b"ATCGN";

/// Options for configuring the FASTA sequence encoder.
///
/// This struct provides configuration options for encoding FASTA sequences,
/// such as which bases to consider during encoding.
///
/// # Fields
///
/// * `bases` - A vector of valid bases (as bytes) to use for encoding. Defaults to "ATCGN".
///
/// # Example
///
/// ```
/// use deepbiop_fa::encode::option::EncoderOption;
///
/// let options = EncoderOption::default();
/// ```
#[gen_stub_pyclass]
#[pyclass(module = "deepbiop.fa")]
#[derive(Debug, Builder, Default, Clone, Serialize, Deserialize)]
pub struct EncoderOption {
    #[pyo3(get, set)]
    #[builder(default = "BASES.to_vec()")]
    pub bases: Vec<u8>,
}

#[gen_stub_pymethods]
#[pymethods]
impl EncoderOption {
    #[new]
    #[pyo3(signature = (bases))]
    fn py_new(bases: String) -> Self {
        EncoderOptionBuilder::default()
            .bases(bases.as_bytes().to_vec())
            .build()
            .expect("Failed to build FqEncoderOption from Python arguments.")
    }
}

impl Display for EncoderOption {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "FaEncoderOption {{ bases: {:?} }}", self.bases)
    }
}
