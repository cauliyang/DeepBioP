use ahash::HashMap;
use rayon::prelude::*;
use std::ops::Range;

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
