use ahash::HashMap;
use anyhow::{Error, Result};
use deepbiop_core::error::DPError;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::ops::Range;

/// Split the quality scores by offsets.
pub fn split_qual_by_offsets(target: &[usize], offsets: &[(usize, usize)]) -> Result<Vec<usize>> {
    let res: Vec<usize> = offsets
        .par_iter()
        .map(|(start, end)| {
            if start == end {
                // Special token
                0
            } else {
                (*start..*end).map(|i| target[i]).sum::<usize>() / (end - start)
            }
        })
        .collect();
    Ok(res)
}

/// Vertorize the target region.
#[pyfunction]
pub fn vertorize_target(start: usize, end: usize, length: usize) -> Result<Vec<usize>> {
    if start > end || end > length {
        return Err(Error::from(DPError::TargetRegionInvalid));
    }

    let mut result = vec![0; length];
    result
        .par_iter_mut()
        .take(end)
        .skip(start)
        .for_each(|x| *x = 1);
    Ok(result)
}

pub fn ascii_list2str(ascii_list: &[i64]) -> String {
    ascii_list
        .par_iter()
        .map(|&c| char::from_u32(c as u32).unwrap())
        .collect()
}

pub fn id_list2seq_i64(id_list: &[i64], tables: &HashMap<i64, char>) -> String {
    id_list.par_iter().map(|id| tables[id]).collect()
}

/// find 1s regions in the labels e.g. 00110011100
pub fn get_label_region(labels: &[i8]) -> Vec<Range<usize>> {
    let mut regions = vec![];

    let mut start = 0;
    let mut end = 0;

    for (i, label) in labels.iter().enumerate() {
        if *label == 1 {
            if start == 0 {
                start = i;
            }
            end = i;
        } else if start != 0 {
            regions.push(start..end + 1);
            start = 0;
            end = 0;
        }
    }

    if start != 0 {
        regions.push(start..end + 1);
    }

    regions
}

mod tests {

    #[test]
    fn test_vertorize_target_valid() {
        use super::vertorize_target;
        let start = 3;
        let end = 5;
        let result = vertorize_target(start, end, 6).unwrap();
        assert_eq!(result, vec![0, 0, 0, 1, 1, 0]);

        let rr = vertorize_target(0, 0, 6).unwrap();
        assert_eq!(rr, vec![0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn test_vertorize_target_invalid() {
        use super::vertorize_target;
        let start = 5;
        let end = 0;
        let result = vertorize_target(start, end, 2);
        assert!(result.is_err());
    }
}
