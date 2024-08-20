pub mod chimeric_count;
pub use chimeric_count::*;

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
