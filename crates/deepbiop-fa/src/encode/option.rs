use derive_builder::Builder;
use serde::{Deserialize, Serialize};
use std::fmt::{self, Display, Formatter};

use pyo3::prelude::*;
use pyo3_stub_gen::derive::*;

pub const BASES: &[u8] = b"ATCGN";
pub const QUAL_OFFSET: u8 = 33;

#[gen_stub_pyclass]
#[pyclass(module = "deepbiop.fa")]
#[derive(Debug, Builder, Default, Clone, Serialize, Deserialize)]
pub struct FaEncoderOption {
    #[pyo3(get, set)]
    #[builder(default = "QUAL_OFFSET")]
    pub qual_offset: u8,

    #[pyo3(get, set)]
    #[builder(default = "BASES.to_vec()")]
    pub bases: Vec<u8>,

    #[pyo3(get, set)]
    #[builder(default = "2")]
    pub threads: usize,
}

#[gen_stub_pymethods]
#[pymethods]
impl FaEncoderOption {
    #[new]
    fn py_new(qual_offset: u8, bases: String, threads: Option<usize>) -> Self {
        FaEncoderOptionBuilder::default()
            .qual_offset(qual_offset)
            .bases(bases.as_bytes().to_vec())
            .threads(threads.unwrap_or(2))
            .build()
            .expect("Failed to build FqEncoderOption from Python arguments.")
    }
}

impl Display for FaEncoderOption {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(
            f,
            "FaEncoderOption {{ qual_offset: {}, bases: {:?}}}",
            self.qual_offset, self.bases
        )
    }
}
