//! a structrual variant or gene fusion event
use std::str::FromStr;

use bstr::BString;
use derive_builder::Builder;

use std::fmt;

/// StructuralVariantType
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StructralVariantType {
    Deletion,
    Duplication,
    Inversion,
    Translocation,
    Insertion,
    UNKNOWN,
}

impl FromStr for StructralVariantType {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "deletion" => Ok(Self::Deletion),
            "DEL" => Ok(Self::Deletion),
            "duplication" => Ok(Self::Duplication),
            "DUP" => Ok(Self::Duplication),
            "inversion" => Ok(Self::Inversion),
            "INV" => Ok(Self::Inversion),
            "translocation" => Ok(Self::Translocation),
            "TRA" => Ok(Self::Translocation),
            "insertion" => Ok(Self::Insertion),
            "INS" => Ok(Self::Insertion),
            _ => Ok(Self::UNKNOWN),
        }
    }
}

impl fmt::Display for StructralVariantType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StructralVariantType::Deletion => write!(f, "DEL"),
            StructralVariantType::Duplication => write!(f, "DUP"),
            StructralVariantType::Inversion => write!(f, "INV"),
            StructralVariantType::Translocation => write!(f, "TRA"),
            StructralVariantType::Insertion => write!(f, "INS"),
            StructralVariantType::UNKNOWN => write!(f, "UNKNOWN"),
        }
    }
}

/// A StructuralVariant is a genomic interval defined by a chromosome, a start position and an end position.
#[derive(Debug, Builder, Clone)]
pub struct StructuralVariant {
    pub sv_type: StructralVariantType,
    pub chr: BString,
    pub breakpoint1: usize,
    pub breakpoint2: usize,
}

impl fmt::Display for StructuralVariant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}:{}:{}-{}",
            self.sv_type, self.chr, self.breakpoint1, self.breakpoint2
        )
    }
}

impl FromStr for StructuralVariant {
    type Err = anyhow::Error;

    /// Parse a string into a StructuralVariant. The string should be formatted as
    /// # Example
    /// ```
    /// use deepbiop_utils::sv::StructuralVariant;
    /// let value =  "DEL:chr1:100-200";
    /// let sv: StructuralVariant = value.parse().unwrap();
    /// ```
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        let parts: Vec<&str> = s.split(':').collect();

        if parts.len() != 3 {
            return Err(anyhow::anyhow!("Invalid format"));
        }

        let sv_type = parts[0];
        let chr = parts[1];

        let positions: Vec<&str> = parts[2].split('-').collect();

        if positions.len() != 2 {
            return Err(anyhow::anyhow!("Invalid format"));
        }

        let start: usize = positions[0].parse()?;
        let end: usize = positions[1].parse()?;

        Ok(Self {
            sv_type: StructralVariantType::from_str(sv_type)?,
            chr: chr.into(),
            breakpoint1: start,
            breakpoint2: end,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_structural_variant_type_from_str() {
        let test_cases = vec![
            ("deletion", StructralVariantType::Deletion),
            ("DEL", StructralVariantType::Deletion),
            ("duplication", StructralVariantType::Duplication),
            ("DUP", StructralVariantType::Duplication),
            ("inversion", StructralVariantType::Inversion),
            ("INV", StructralVariantType::Inversion),
            ("translocation", StructralVariantType::Translocation),
            ("TRA", StructralVariantType::Translocation),
            ("insertion", StructralVariantType::Insertion),
            ("INS", StructralVariantType::Insertion),
            ("unknown", StructralVariantType::UNKNOWN),
        ];

        for (input, _expected) in test_cases {
            let result = StructralVariantType::from_str(input).unwrap();
            assert!(matches!(result, _expected));
        }
    }

    #[test]
    fn test_structural_variant_from_str() {
        let test_cases = vec![
            (
                "DEL:chr1:100-200",
                StructuralVariant {
                    sv_type: StructralVariantType::Deletion,
                    chr: "chr1".into(),
                    breakpoint1: 100,
                    breakpoint2: 200,
                },
            ),
            (
                "duplication:chr2:300-400",
                StructuralVariant {
                    sv_type: StructralVariantType::Duplication,
                    chr: "chr2".into(),
                    breakpoint1: 300,
                    breakpoint2: 400,
                },
            ),
            (
                "INV:chrX:1000-2000",
                StructuralVariant {
                    sv_type: StructralVariantType::Inversion,
                    chr: "chrX".into(),
                    breakpoint1: 1000,
                    breakpoint2: 2000,
                },
            ),
        ];

        for (input, expected) in test_cases {
            let result: StructuralVariant = input.parse().unwrap();
            assert_eq!(result.sv_type.clone() as i32, expected.sv_type as i32);
            assert_eq!(result.chr, expected.chr);
            assert_eq!(result.breakpoint1, expected.breakpoint1);
            assert_eq!(result.breakpoint2, expected.breakpoint2);
        }
    }

    #[test]
    fn test_structural_variant_from_str_invalid_format() {
        let invalid_inputs = vec![
            "invalid_format",
            "DEL:chr1",
            "DEL:chr1:100",
            "DEL:chr1:abc-200",
            "DEL:chr1:100-def",
        ];

        for input in invalid_inputs {
            assert!(StructuralVariant::from_str(input).is_err());
        }
    }

    #[test]
    fn test_structural_variant_builder() {
        let sv = StructuralVariantBuilder::default()
            .sv_type(StructralVariantType::Deletion)
            .chr("chr1".into())
            .breakpoint1(100)
            .breakpoint2(200)
            .build()
            .unwrap();

        assert!(matches!(sv.sv_type, StructralVariantType::Deletion));
        assert_eq!(sv.chr, "chr1");
        assert_eq!(sv.breakpoint1, 100);
        assert_eq!(sv.breakpoint2, 200);

        let sv2 = sv.clone();

        assert_eq!(sv.sv_type, sv2.sv_type);
    }
}
