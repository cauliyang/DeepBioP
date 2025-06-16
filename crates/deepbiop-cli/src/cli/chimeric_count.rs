use anyhow::Result;
use clap::Parser;
use deepbiop_bam as bam;
use std::{
    fs::File,
    io::{BufWriter, Write},
    path::PathBuf,
};

use super::set_up_threads;

#[derive(Debug, Parser)]
pub struct CountChimeric {
    /// path to the bam file
    #[arg(value_name = "bam", action=clap::ArgAction::Append)]
    bam: Vec<PathBuf>,

    /// output name of chimeric reads to a file
    #[arg(short, long, default_value = "false")]
    read_name: bool,

    /// threads number
    #[arg(short, long, default_value = "2")]
    threads: Option<usize>,
}

impl CountChimeric {
    pub fn run(&self) -> Result<()> {
        set_up_threads(self.threads)?;

        if !self.read_name {
            let res = bam::chimeric::count_chimeric_reads_for_paths(&self.bam, self.threads);
            for (path, count) in res {
                log::info!("{}: {}", path.to_string_lossy(), count);
            }
        } else {
            let res =
                bam::chimeric::extract_chimeric_reads_name_for_paths(&self.bam, self.threads)?;

            for (path, names) in res {
                let file_name = format!("{}.chimeric_reads.txt", path.to_string_lossy());

                log::info!("{}: {}", path.to_string_lossy(), names.len());
                log::info!("Writing chimeric reads name to {}", file_name);

                let file = File::create(file_name)?;
                let mut writer = BufWriter::new(file);
                for name in names {
                    writeln!(writer, "{}", name)?;
                }
            }
        }
        Ok(())
    }
}
