use deepbiop_core::default::{BASES, QUAL_OFFSET};
use derive_builder::Builder;
use serde::{Deserialize, Serialize};
use std::fmt::{self, Display, Formatter};

use pyo3::prelude::*;
use pyo3_stub_gen::derive::*;

#[gen_stub_pyclass]
#[pyclass(module = "deepbiop.fq")]
#[derive(Debug, Builder, Default, Clone, Serialize, Deserialize)]
pub struct EncoderOption {
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
impl EncoderOption {
    #[new]
    #[pyo3(signature = (qual_offset, bases, threads=None))]
    fn py_new(qual_offset: u8, bases: String, threads: Option<usize>) -> Self {
        EncoderOptionBuilder::default()
            .qual_offset(qual_offset)
            .bases(bases.as_bytes().to_vec())
            .threads(threads.unwrap_or(2))
            .build()
            .expect("Failed to build FqEncoderOption from Python arguments.")
    }
}

impl Display for EncoderOption {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(
            f,
            "FqEncoderOption {{ qual_offset: {}, bases: {:?}}}",
            self.qual_offset, self.bases,
        )
    }
}
