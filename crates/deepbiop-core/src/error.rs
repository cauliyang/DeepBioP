use pyo3::PyErr;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum DPError {
    #[error("An error occurred: {0}")]
    Generic(String),

    #[error("Another error occurred")]
    Another,

    #[error("The sequence is shorter than the k-mer size")]
    SeqShorterThanKmer,

    #[error("The target region is invalid")]
    TargetRegionInvalid,

    #[error("The k-mer id is invalid")]
    InvalidKmerId,

    #[error("The interval is invalid: {0}")]
    InvalidInterval(String),

    #[error("The sequence and quality scores have different lengths: {0}")]
    NotSameLengthForQualityAndSequence(String),
}

impl From<DPError> for PyErr {
    fn from(error: DPError) -> PyErr {
        use DPError::*;
        match error {
            Generic(message) => pyo3::exceptions::PyException::new_err(message),
            Another => pyo3::exceptions::PyException::new_err("Another error occurred"),
            SeqShorterThanKmer => pyo3::exceptions::PyException::new_err(
                "The sequence is shorter than the k-mer size",
            ),
            TargetRegionInvalid => {
                pyo3::exceptions::PyException::new_err("The target region is invalid")
            }
            InvalidKmerId => pyo3::exceptions::PyException::new_err("The k-mer id is invalid"),
            InvalidInterval(interval) => pyo3::exceptions::PyException::new_err(format!(
                "The interval is invalid: {:?}",
                interval
            )),
            NotSameLengthForQualityAndSequence(mes) => {
                pyo3::exceptions::PyException::new_err(format!(
                    "The sequence and quality scores have different lengths: {:?}",
                    mes
                ))
            }
        }
    }
}
