use noodles::sam::alignment::record::Cigar as CigarTrait;
use noodles::sam::record::Cigar;
use std::path::Path;
use std::{fs::File, num::NonZeroUsize, thread};

use anyhow::Result;
use bstr::BString;
use noodles::{bam, bgzf};
use rayon::prelude::*;

use noodles::sam::alignment::record::data::field::Tag;
use noodles::sam::alignment::record::data::field::Value;

use deepbiop_utils::interval::GenomicInterval;
use deepbiop_utils::interval::GenomicIntervalBuilder;

use derive_builder::Builder;
use pyo3::prelude::*;
use std::str::FromStr;

#[pyclass]
#[derive(Debug, Builder)]
pub struct ChimericEvent {
    pub name: BString,
    pub intervals: Vec<GenomicInterval>,
}

impl ChimericEvent {
    /// Parse sa tag string into a ChimericEvent.
    /// The string should be formatted as `rname,pos,strand,CIGAR,mapQ,NM;`
    /// # Example
    /// ```
    /// use deepbiop_bam::chimeric::ChimericEvent;
    /// let  value =  "chr1,100,+,100M,60,0;chr2,200,+,100M,60,0";
    /// let chimeric_event: ChimericEvent = value.parse().unwrap();
    /// ```
    pub fn parse_sa_tag(sa_tag: &str) -> Result<Self> {
        let mut res = vec![];
        let mut name = "";

        for sa in sa_tag.split(';') {
            let mut splits = sa.split(',');
            let sa_reference_name = splits.next().unwrap();
            let sa_start = splits.next().unwrap().parse::<usize>()?;
            let _sa_strand = splits.next().unwrap();
            let sa_cigar = splits.next().unwrap();
            let _sa_mapq = splits.next().unwrap();
            let _sa_nm = splits.next().unwrap();

            name = sa_reference_name;
            let sa_end = sa_start + Cigar::new(sa_cigar.as_bytes()).alignment_span().unwrap();

            let sa_interval = GenomicIntervalBuilder::default()
                .chr(sa_reference_name.into())
                .start(sa_start)
                .end(sa_end)
                .build()?;
            res.push(sa_interval);
        }

        Ok(ChimericEventBuilder::default()
            .name(name.into())
            .intervals(res)
            .build()
            .unwrap())
    }
}

impl FromStr for ChimericEvent {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        ChimericEvent::parse_sa_tag(s)
    }
}

pub fn create_chimeric_events_from_chimeric_reads<P: AsRef<Path>>(
    bam: P,
    threads: Option<usize>,
) -> Result<Vec<ChimericEvent>> {
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

    let res: Result<Vec<ChimericEvent>> = reader
        .records()
        .par_bridge()
        .filter_map(|result| {
            let record = result.unwrap();
            let is_mapped = !record.flags().is_unmapped();
            let is_not_secondary = !record.flags().is_secondary();
            let is_primary = !record.flags().is_supplementary();

            if is_primary && is_mapped && is_not_secondary {
                return Some(record);
            }
            None
        })
        .map(|record| {
            let reference_id = record.reference_sequence_id().unwrap().unwrap();
            // get the reference name
            let reference_name = references.get_index(reference_id).unwrap().0;
            let reference_start = usize::from(record.alignment_start().unwrap().unwrap());
            let reference_end = reference_start + record.cigar().alignment_span().unwrap();

            let interval = GenomicIntervalBuilder::default()
                .chr(reference_name.clone())
                .start(reference_start)
                .end(reference_end)
                .build()?;

            let mut chimeric_event = ChimericEventBuilder::default()
                .name(reference_name.clone())
                .intervals(vec![interval])
                .build()?;

            if let Some(Ok(Value::String(sa_string))) = record.data().get(&Tag::OTHER_ALIGNMENTS) {
                // has sa tag
                let sa_string = sa_string.to_string();
                let sa_chimeric_event: ChimericEvent = sa_string.as_str().parse()?;
                chimeric_event.intervals.extend(sa_chimeric_event.intervals)
            }

            Ok(chimeric_event)
        })
        .collect();

    res
}
