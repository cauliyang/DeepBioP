use std::fs::File;
use std::io::prelude::*;
use std::io::BufReader;
use std::io::Write;
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;

use ahash::HashMap;
use ahash::HashMapExt;
use anyhow::Result;
use derive_builder::Builder;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use tempfile::tempdir;

use pyo3_stub_gen::derive::*;

pub const MIN_SEQ_SIZE: usize = 20;
// psLayout version 3

// match   mis-    rep.    N's     Q gap   Q gap   T gap   T gap   strand  Q               Q       Q       Q       T               T       T    T        block   blockSizes      qStarts  tStarts
//         match   match           count   bases   count   bases           name            size    start   end     name            size    startend      count
// ---------------------------------------------------------------------------------------------------------------------------------------------------------------
// 23      1       0       0       0       0       0       0       +       seq     51      3       27      chr12   133275309       11447342     11447366 1       24,     3,      11447342,

#[gen_stub_pyclass]
#[pyclass(module = "deepbiop.utils")]
#[derive(Debug, Default, Builder, Clone, Serialize, Deserialize)]
pub struct PslAlignment {
    #[pyo3(get, set)]
    pub qname: String,
    #[pyo3(get, set)]
    pub qsize: usize,
    #[pyo3(get, set)]
    pub qstart: usize,
    #[pyo3(get, set)]
    pub qend: usize,
    #[pyo3(get, set)]
    pub qmatch: usize,
    #[pyo3(get, set)]
    pub tname: String,
    #[pyo3(get, set)]
    pub tsize: usize,
    #[pyo3(get, set)]
    pub tstart: usize,
    #[pyo3(get, set)]
    pub tend: usize,
    #[pyo3(get, set)]
    pub identity: f32,
}

pub fn parse_psl_by_qname<P: AsRef<Path>>(file: P) -> Result<HashMap<String, Vec<PslAlignment>>> {
    let result = parse_psl(file)?;
    Ok(result.into_iter().fold(HashMap::new(), |mut acc, al| {
        let qname = al.qname.clone();
        acc.entry(qname).or_default().push(al);
        acc
    }))
}

pub fn parse_psl<P: AsRef<Path>>(file: P) -> Result<Vec<PslAlignment>> {
    let file = File::open(file)?;
    let mut reader = BufReader::new(file);
    let mut line = String::new();

    // Skip the first 5 lines
    for _ in 0..5 {
        reader.read_line(&mut line)?;
        line.clear();
    }
    let mut alignments = Vec::new();

    // only get match, Qsize, qstart, qend, Tsize, tstart, tend
    while reader.read_line(&mut line)? > 0 {
        let fields: Vec<&str> = line.split_whitespace().collect();

        let match_: usize = lexical::parse(fields[0])?;
        let qname = fields[9];
        let qsize: usize = lexical::parse(fields[10])?;
        let qstart: usize = lexical::parse(fields[11])?;
        let qend: usize = lexical::parse(fields[12])?;

        let tname = fields[13];
        let tsize: usize = lexical::parse(fields[14])?;
        let tstart: usize = lexical::parse(fields[15])?;
        let tend: usize = lexical::parse(fields[16])?;

        let identity = match_ as f32 / qsize as f32;

        let al = PslAlignmentBuilder::default()
            .qname(qname.to_string())
            .qsize(qsize)
            .qstart(qstart)
            .qend(qend)
            .qmatch(match_)
            .tname(tname.to_string())
            .tsize(tsize)
            .tstart(tstart)
            .tend(tend)
            .identity(identity)
            .build()?;
        alignments.push(al);
        line.clear();
    }
    Ok(alignments)
}

pub fn blat_for_seq<P: AsRef<Path> + AsRef<std::ffi::OsStr>>(
    path: P,
    blat_cli: P,
    two_bit: P,
    output: P,
) -> Result<Vec<PslAlignment>> {
    let _output = Command::new(blat_cli)
        .arg("-stepSize=5")
        .arg("-repMatch=2253")
        .arg("-minScore=20")
        .arg("-minIdentity=0")
        .arg(two_bit)
        .arg(path)
        .arg(&output)
        .output()?;

    if !_output.status.success() {
        return Err(anyhow::anyhow!(
            "blat failed: {}",
            String::from_utf8_lossy(&_output.stderr)
        ));
    }
    parse_psl(output)
}

// ./blat -stepSize=5 -repMatch=2253 -minScore=20 -minIdentity=0  hg38.2bit t.fa  output.psl
pub fn blat<P: AsRef<Path> + AsRef<std::ffi::OsStr>>(
    seq: &str,
    blat_cli: P,
    two_bit: P,
    output: Option<&str>,
) -> Result<Vec<PslAlignment>> {
    log::debug!("blat_cli: {}", seq);

    // Create a file inside `std::env::temp_dir()`.
    let dir = tempdir()?;
    let file1 = dir.path().join("seq.fa");
    let mut tmp_file1 = File::create(file1.clone())?;
    writeln!(tmp_file1, ">seq\n")?;
    writeln!(tmp_file1, "{}", seq)?;

    // Create a directory inside `std::env::temp_dir()`.
    let output_file = if let Some(value) = output {
        PathBuf::from(value)
    } else {
        dir.path().join("output.psl")
    };

    let _output = Command::new(blat_cli)
        .arg("-stepSize=5")
        .arg("-repMatch=2253")
        .arg("-minScore=20")
        .arg("-minIdentity=0")
        .arg(two_bit)
        .arg(file1)
        .arg(output_file.clone())
        .output()?;

    if !_output.status.success() || !output_file.exists() {
        return Err(anyhow::anyhow!(
            "blat failed: {}",
            String::from_utf8_lossy(&_output.stderr)
        ));
    }

    parse_psl(output_file)
}

#[cfg(test)]
mod tests {
    // use super::*;

    #[test]
    fn test_parse_psl() {
        let _file = "tests/data/sample.psl";
    }
}
