use crate::encode;
use pyo3::prelude::*;

#[pymethods]
impl encode::TensorEncoder {
    #[new]
    fn py_new(
        option: encode::FqEncoderOption,
        tensor_max_width: Option<usize>,
        tensor_max_seq_len: Option<usize>,
    ) -> Self {
        encode::TensorEncoder::new(option, tensor_max_width, tensor_max_seq_len)
    }
}

#[pymethods]
impl encode::JsonEncoder {
    #[new]
    fn py_new(option: encode::FqEncoderOption) -> Self {
        encode::JsonEncoder::new(option)
    }
}

#[pymethods]
impl encode::ParquetEncoder {
    #[new]
    fn py_new(option: encode::FqEncoderOption) -> Self {
        encode::ParquetEncoder::new(option)
    }
}
