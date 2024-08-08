use anyhow::Result;
use derive_builder::Builder;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

use super::traits::Overlap;

/// A segment is a genomic interval defined by a chromosome, a start position and an end position.
/// The start position is inclusive and the end position is exclusive.
#[pyclass]
#[derive(Debug, Builder, Clone, Deserialize, Serialize, PartialEq)]
pub struct Segment {
    #[pyo3(get, set)]
    pub chr: String,
    #[pyo3(get, set)]
    pub start: usize,
    #[pyo3(get, set)]
    pub end: usize,
}

// impl SegmentBuilder {
//     fn validate(&self) -> Result<(), String> {
//         if self.start > self.end {
//             Err("start must be less than end".to_string())
//         } else {
//             Ok(())
//         }
//     }
// }

#[pymethods]
impl Segment {
    #[new]
    fn py_new(chr: &str, start: usize, end: usize) -> Self {
        Segment {
            chr: chr.to_string(),
            start,
            end,
        }
    }

    #[pyo3(name = "overlap")]
    fn py_overlap(&self, other: &Segment) -> bool {
        self.overlap(other)
    }

    fn __repr__(&self) -> String {
        format!(
            "Segment(chr={}, start={}, end={})",
            self.chr, self.start, self.end
        )
    }
}

impl Segment {
    pub fn new(chr: &str, start: usize, end: usize) -> Result<Self> {
        if start > end {
            Err(anyhow::anyhow!("start must be less than end"))
        } else {
            Ok(Self {
                chr: chr.to_string(),
                start,
                end,
            })
        }
    }
}

impl Overlap for Segment {
    fn overlap(&self, other: &Self) -> bool {
        self.chr == other.chr && self.start < other.end && self.end > other.start
    }
}

impl PartialOrd for Segment {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        if self.chr == other.chr {
            if self.start == other.start {
                self.end.partial_cmp(&other.end)
            } else {
                self.start.partial_cmp(&other.start)
            }
        } else {
            self.chr.partial_cmp(&other.chr)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_segment() {
        let segment = SegmentBuilder::default()
            .chr("chr1".to_string())
            .start(100)
            .end(200)
            .build()
            .unwrap();

        let segment2 = SegmentBuilder::default()
            .chr("chr1".to_string())
            .start(150)
            .end(250)
            .build()
            .unwrap();

        assert!(segment.overlap(&segment2));

        // let segment3 = SegmentBuilder::default()
        //     .chr("chr2".to_string())
        //     .start(350)
        //     .end(250)
        //     .build();
        // assert!(segment3.is_err());

        let segment4 = Segment::new("chr2", 100, 200).unwrap();

        assert!(!segment.overlap(&segment4));
    }
}
