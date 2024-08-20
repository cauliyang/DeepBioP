use clap::Parser;
use std::path::PathBuf;

#[derive(Debug, Parser)]
pub struct CountChimeric {
    /// path to the bam file
    #[arg(value_name = "bam", action=clap::ArgAction::Append)]
    bam: Vec<PathBuf>,
}
