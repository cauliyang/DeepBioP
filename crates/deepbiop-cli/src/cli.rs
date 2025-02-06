pub mod bam2fq;
pub use bam2fq::*;
pub mod fq2fa;
pub use fq2fa::*;
pub mod fa2fq;
pub use fa2fq::*;
pub mod chimeric_count;
pub use chimeric_count::*;
pub mod fx2parquet;
pub use fx2parquet::*;
pub mod extractfx;
pub use extractfx::*;
pub mod fxs2one;
pub use fxs2one::*;
pub mod countfx;
pub use countfx::*;

use anyhow::Result;
// Set up threads only once, using the common_opts from the top-level Cli struct
pub fn set_up_threads(threads: Option<usize>) -> Result<()> {
    log::info!("Threads number: {:?}", threads.unwrap());

    rayon::ThreadPoolBuilder::new()
        .num_threads(threads.unwrap())
        .build_global()
        .unwrap();
    Ok(())
}
