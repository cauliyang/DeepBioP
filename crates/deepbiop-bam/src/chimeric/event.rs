use noodles::sam::alignment::record::Cigar as CigarTrait;
use noodles::sam::record::Cigar;
use std::path::Path;
use std::{fs::File, num::NonZeroUsize, thread};

use anyhow::Result;
use bstr::BString;
use noodles::{bam, bgzf, sam};
use rayon::prelude::*;

use noodles::sam::alignment::record::data::field::Tag;
use noodles::sam::alignment::record::data::field::Value;

use deepbiop_utils::interval::GenomicInterval;
use deepbiop_utils::interval::GenomicIntervalBuilder;

use derive_builder::Builder;
use log::debug;
use pyo3::prelude::*;
use std::str::FromStr;

use super::{is_chimeric_record, is_retain_record};

/// A chimeric event.
#[pyclass]
#[derive(Debug, Builder)]
pub struct ChimericEvent {
    /// The name of the chimeric event.
    pub name: Option<BString>,
    /// The intervals of the chimeric event.
    pub intervals: Vec<GenomicInterval>,
}

impl ChimericEvent {
    /// Get the length of the chimeric event.
    pub fn len(&self) -> usize {
        self.intervals.len()
    }

    /// Check if the chimeric event is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Parse sa tag string into a ChimericEvent.
    /// The string should be formatted as `rname,pos,strand,CIGAR,mapQ,NM;`
    ///
    /// # Example
    /// ```
    /// use deepbiop_bam as bam;
    /// use bam::chimeric::ChimericEvent;
    /// let  value =  "chr1,100,+,100M,60,0;chr2,200,+,100M,60,0";
    /// let chimeric_event: ChimericEvent = ChimericEvent::parse_sa_tag(value, None).unwrap();
    /// let value2 = "chr8,109336127,+,155S308M7054D449S,60,3;";
    /// let chimeric_event2: ChimericEvent = ChimericEvent::parse_sa_tag(value2, Some("value2")).unwrap();
    /// assert_eq!(chimeric_event.len(),2);
    /// ```
    pub fn parse_sa_tag(sa_tag: &str, name: Option<&str>) -> Result<Self> {
        debug!("Parsing sa tag: {}", sa_tag);

        let mut res = vec![];

        for sa in sa_tag.split_terminator(';') {
            let mut splits = sa.split(',');

            let sa_reference_name = splits.next().unwrap();
            let sa_start: usize = lexical::parse(splits.next().unwrap())?;
            let _sa_strand = splits.next().unwrap();
            let sa_cigar = splits.next().unwrap();
            let _sa_mapq = splits.next().unwrap();
            let _sa_nm = splits.next().unwrap();

            let sa_end = sa_start + Cigar::new(sa_cigar.as_bytes()).alignment_span()?;

            let sa_interval = GenomicIntervalBuilder::default()
                .chr(sa_reference_name.into())
                .start(sa_start)
                .end(sa_end)
                .build()?;
            res.push(sa_interval);
        }

        Ok(ChimericEventBuilder::default()
            .name(name.as_ref().map(|&x| x.into()))
            .intervals(res)
            .build()?)
    }

    /// Construct a ChimericEvent from a noodle bam record.
    pub fn parse_noodle_bam_record(
        record: &bam::Record,
        references: &sam::header::ReferenceSequences,
    ) -> Result<Self> {
        let reference_id = record.reference_sequence_id().unwrap()?;
        // get the reference name
        let reference_name = references.get_index(reference_id).unwrap().0;
        let reference_start = usize::from(record.alignment_start().unwrap()?);
        let reference_end = reference_start + record.cigar().alignment_span()?;

        let record_name = record.name().unwrap();

        let interval = GenomicIntervalBuilder::default()
            .chr(reference_name.clone())
            .start(reference_start)
            .end(reference_end)
            .build()?;

        let mut chimeric_event = ChimericEventBuilder::default()
            .name(Some(record_name.into()))
            .intervals(vec![interval])
            .build()?;

        if let Some(Ok(Value::String(sa_string))) = record.data().get(&Tag::OTHER_ALIGNMENTS) {
            // has sa tag
            let sa_string = sa_string.to_string();
            let sa_chimeric_event: ChimericEvent = sa_string.as_str().parse()?;
            chimeric_event.intervals.extend(sa_chimeric_event.intervals)
        }

        Ok(chimeric_event)
    }

    /// Construct a ChimericEvent from a string chr:start-end,chr:start-end
    /// # Example
    /// ```
    /// use deepbiop_bam as bam;
    /// use bam::chimeric::ChimericEvent;
    /// let  value =  "chr1:103959-104483,chr1:280386-280637";
    /// let chimeric_event: ChimericEvent = ChimericEvent::parse_list_pos(value, "value").unwrap();
    /// assert_eq!(chimeric_event.len(),2);
    /// ```
    pub fn parse_list_pos(s: &str, name: &str) -> Result<Self> {
        let intervals = s
            .par_split(',')
            .map(|event| {
                let mut splits = event.split(':');
                let chr = splits.next().unwrap();
                let positions: Vec<&str> = splits.next().unwrap().split('-').collect();
                let start: usize = lexical::parse(positions[0]).unwrap();
                let end: usize = lexical::parse(positions[1]).unwrap();
                GenomicIntervalBuilder::default()
                    .chr(chr.into())
                    .start(start)
                    .end(end)
                    .build()
                    .unwrap()
            })
            .collect::<Vec<GenomicInterval>>();

        Ok(ChimericEventBuilder::default()
            .name(Some(name.into()))
            .intervals(intervals)
            .build()?)

        // Ok(res)
    }
}

impl FromStr for ChimericEvent {
    type Err = anyhow::Error;

    /// # Example
    /// ```
    /// use deepbiop_bam as bam;
    /// use bam::chimeric::ChimericEvent;
    /// let  value =  "chr1,100,+,100M,60,0;chr2,200,+,100M,60,0";
    /// let chimeric_event: ChimericEvent = value.parse().unwrap();
    /// let value2 = "chr8,109336127,+,155S308M7054D449S,60,3;";
    /// let chimeric_event2: ChimericEvent = value2.parse().unwrap();
    /// ```
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        ChimericEvent::parse_sa_tag(s, None)
    }
}

/// Create a list of chimeric events from a bam file.
pub fn create_chimeric_events_from_bam<P, F>(
    bam: P,
    threads: Option<usize>,
    predict: Option<F>,
) -> Result<Vec<ChimericEvent>>
where
    F: Fn(&bam::Record) -> bool + std::marker::Sync,
    P: AsRef<Path>,
{
    let worker_count = if let Some(threads) = threads {
        std::num::NonZeroUsize::new(threads)
            .unwrap()
            .min(thread::available_parallelism().unwrap_or(NonZeroUsize::MIN))
    } else {
        thread::available_parallelism().unwrap_or(NonZeroUsize::MIN)
    };

    let file = File::open(bam.as_ref())?;
    let decoder = bgzf::MultithreadedReader::with_worker_count(worker_count, file);
    let mut reader = bam::io::Reader::from(decoder);
    let header = reader.read_header()?;
    let references = header.reference_sequences();

    reader
        .records()
        .par_bridge()
        .filter_map(|result| {
            let record = result.unwrap();
            if is_retain_record(&record) && is_chimeric_record(&record) {
                if let Some(predict_function) = &predict {
                    if predict_function(&record) {
                        Some(record)
                    } else {
                        None
                    }
                } else {
                    Some(record)
                }
            } else {
                None
            }
        })
        .map(|record| ChimericEvent::parse_noodle_bam_record(&record, references))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_chimeric_events_from_bam() {
        // Create a test BAM file
        let bam = "tests/data/test_chimric_reads.bam";

        // Define a predict function for testing
        let predict_fn = |_record: &bam::Record| -> bool {
            // Define predict function logic here for testing
            true
        };
        // Call the function with test parameters
        let result = create_chimeric_events_from_bam(bam, Some(2), Some(predict_fn));
        // Assert on the result
        assert!(result.is_ok());
        let chimeric_events = result.unwrap();
        assert_eq!(chimeric_events.len(), 100);
    }
}
