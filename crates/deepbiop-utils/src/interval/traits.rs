use std::ops::Range;

pub trait Overlap {
    fn overlap(&self, other: &Self) -> bool;
}

impl<P: PartialOrd> Overlap for Range<P> {
    fn overlap(&self, other: &Self) -> bool {
        self.start < other.end && self.end > other.start
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_overlap_for_range() {
        let a = 1..10;
        let b = 5..15;
        assert!(a.overlap(&b));

        let c = 10..15;
        let d = 15..20;
        assert!(!c.overlap(&d));
    }
}
