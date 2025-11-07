use pyo3::PyErr;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum DPError {
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

    #[error("Invalid alphabet character '{character}' at position {position} in sequence. Expected one of: {expected}")]
    InvalidAlphabet {
        character: char,
        position: usize,
        expected: String,
    },

    #[error("Quality score mismatch: sequence length ({seq_len}) != quality length ({qual_len})")]
    QualityMismatch { seq_len: usize, qual_len: usize },

    #[error("Invalid CIGAR string: {message}")]
    InvalidCigar { message: String },

    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    #[error("Invalid value: {0}")]
    InvalidValue(String),
}

impl From<DPError> for PyErr {
    fn from(error: DPError) -> PyErr {
        use DPError::*;
        match error {
            SeqShorterThanKmer => pyo3::exceptions::PyValueError::new_err(
                "The sequence is shorter than the k-mer size",
            ),
            TargetRegionInvalid => {
                pyo3::exceptions::PyValueError::new_err("The target region is invalid")
            }
            InvalidKmerId => pyo3::exceptions::PyValueError::new_err("The k-mer id is invalid"),
            InvalidInterval(interval) => pyo3::exceptions::PyValueError::new_err(format!(
                "The interval is invalid: {}",
                interval
            )),
            NotSameLengthForQualityAndSequence(mes) => {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "The sequence and quality scores have different lengths: {}",
                    mes
                ))
            }
            InvalidAlphabet {
                character,
                position,
                expected,
            } => pyo3::exceptions::PyValueError::new_err(format!(
                "Invalid alphabet character '{}' at position {}. Expected one of: {}",
                character, position, expected
            )),
            QualityMismatch { seq_len, qual_len } => {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "Quality score mismatch: sequence length ({}) != quality length ({})",
                    seq_len, qual_len
                ))
            }
            InvalidCigar { message } => pyo3::exceptions::PyValueError::new_err(format!(
                "Invalid CIGAR string: {}",
                message
            )),
            InvalidParameter(msg) => {
                pyo3::exceptions::PyValueError::new_err(format!("Invalid parameter: {}", msg))
            }
            InvalidValue(msg) => {
                pyo3::exceptions::PyValueError::new_err(format!("Invalid value: {}", msg))
            }
        }
    }
}
