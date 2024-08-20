//! Cigar operations.

use anyhow::Result;

use noodles::sam::alignment::record::cigar::op::{Kind, Op};
use noodles::sam::record::Cigar;

/// Convert a vector of cigar operations to a string.
pub fn cigar_to_string(cigar: &[Op]) -> Result<String> {
    let mut cigar_str = String::new();

    for op in cigar {
        let kind_str = match op.kind() {
            Kind::Match => "M",
            Kind::Insertion => "I",
            Kind::Deletion => "D",
            Kind::Skip => "N",
            Kind::SoftClip => "S",
            Kind::HardClip => "H",
            Kind::Pad => "P",
            Kind::SequenceMatch => "=",
            Kind::SequenceMismatch => "X",
        };
        cigar_str.push_str(&format!("{}{}", op.len(), kind_str));
    }

    Ok(cigar_str)
}

fn _calc_softclips(cigars: &[Op]) -> Result<(usize, usize)> {
    let len = cigars.len();

    // Calculate leading soft clips
    let left_softclips = if len > 0 && cigars[0].kind() == Kind::SoftClip {
        cigars[0].len()
    } else if len > 1 && cigars[0].kind() == Kind::HardClip && cigars[1].kind() == Kind::SoftClip {
        cigars[1].len()
    } else {
        0
    };

    // Calculate trailing soft clips
    let right_softclips = if len > 0 && cigars[len - 1].kind() == Kind::SoftClip {
        cigars[len - 1].len()
    } else if len > 1
        && cigars[len - 1].kind() == Kind::HardClip
        && cigars[len - 2].kind() == Kind::SoftClip
    {
        cigars[len - 2].len()
    } else {
        0
    };

    Ok((left_softclips, right_softclips))
}

/// Calculate the number of leading and trailing soft clips in a cigar string.
pub fn calc_softclips(cigar: &Cigar) -> Result<(usize, usize)> {
    let ops: Vec<Op> = cigar.iter().collect::<Result<Vec<_>, _>>()?;
    _calc_softclips(&ops)
}
