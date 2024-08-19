use anyhow::Result;
use derive_builder::Builder;
use pyo3::prelude::*;

use super::traits::Overlap;

use bstr::BString;
use std::str::FromStr;

use pyo3_stub_gen::derive::*;

/// A segment is a genomic interval defined by a chromosome, a start position and an end position.
/// The start position is inclusive and the end position is exclusive.
#[gen_stub_pyclass]
#[pyclass(module = "deepbiop.utils")]
#[derive(Debug, Builder, Clone, PartialEq)]
#[builder(build_fn(validate = "Self::validate"))]
pub struct GenomicInterval {
    pub chr: BString,
    #[pyo3(get, set)]
    pub start: usize,
    #[pyo3(get, set)]
    pub end: usize,
}

impl FromStr for GenomicInterval {
    type Err = anyhow::Error;

    /// Parse a string into a GenomicInterval. The string should be formatted as
    /// # Example
    /// ```
    /// use deepbiop_utils::interval::GenomicInterval;
    /// let  value =  "chr1:100-200";
    /// let interval: GenomicInterval = value.parse().unwrap();
    /// ```
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        let parts: Vec<&str> = s.split(':').collect();
        let chr = parts[0];
        let positions: Vec<&str> = parts[1].split('-').collect();
        let start: usize = positions[0].parse()?;
        let end: usize = positions[1].parse()?;

        Ok(Self {
            chr: chr.into(),
            start,
            end,
        })
    }
}

impl GenomicIntervalBuilder {
    fn validate(&self) -> Result<(), String> {
        if self.start > self.end {
            Err("start must be less than end".to_string())
        } else {
            Ok(())
        }
    }
}

impl GenomicInterval {
    pub fn new(chr: &str, start: usize, end: usize) -> Result<Self> {
        if start > end {
            Err(anyhow::anyhow!("start must be less than end"))
        } else {
            Ok(Self {
                chr: chr.into(),
                start,
                end,
            })
        }
    }
}

impl Overlap for GenomicInterval {
    fn overlap(&self, other: &Self) -> bool {
        self.chr == other.chr && self.start < other.end && self.end > other.start
    }
}

impl PartialOrd for GenomicInterval {
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
        let segment = GenomicIntervalBuilder::default()
            .chr("chr1".into())
            .start(100)
            .end(200)
            .build()
            .unwrap();

        let segment2 = GenomicIntervalBuilder::default()
            .chr("chr1".into())
            .start(150)
            .end(250)
            .build()
            .unwrap();

        assert!(segment.overlap(&segment2));

        let segment3 = GenomicIntervalBuilder::default()
            .chr("chr2".into())
            .start(350)
            .end(250)
            .build();
        assert!(segment3.is_err());

        let segment4 = GenomicInterval::new("chr2", 100, 200).unwrap();

        assert!(!segment.overlap(&segment4));
    }
}
