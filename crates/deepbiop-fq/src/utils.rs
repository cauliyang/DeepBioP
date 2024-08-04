use std::ops::Range;

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
