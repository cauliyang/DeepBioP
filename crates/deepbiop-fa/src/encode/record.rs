use bstr::BString;
use derive_builder::Builder;

#[derive(Debug, Default, Builder)]
pub struct RecordData {
    pub id: BString,
    pub seq: BString,
}

impl RecordData {
    pub fn new(id: BString, seq: BString) -> Self {
        Self { id, seq }
    }
}

impl From<(Vec<u8>, Vec<u8>)> for RecordData {
    fn from(data: (Vec<u8>, Vec<u8>)) -> Self {
        Self::new(data.0.into(), data.1.into())
    }
}

impl From<(BString, BString)> for RecordData {
    fn from(data: (BString, BString)) -> Self {
        Self::new(data.0, data.1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_record_data_build() {
        let record = RecordDataBuilder::default()
            .id("id".into())
            .seq("seq".into())
            .build()
            .unwrap();

        assert_eq!(record.id, "id");
        assert_eq!(record.seq, "seq");
    }
}
