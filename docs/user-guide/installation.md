# Installation

Deepbiop is a library and installation is as simple as invoking the package manager of the corresponding programming language.

=== ":fontawesome-brands-python: Python"

    ```bash
    pip install deepbiop
    ```

=== ":fontawesome-brands-rust: Rust"

    ```shell
    cargo add polars -F fastq

    # Or Cargo.toml
    [dependencies]
    deepbiop = { version = "x", features = ["fastq", ...]}
    ```

## Importing

To use the library import it into your project

=== ":fontawesome-brands-python: Python"

    ```python
    import deepbiop as dp
    ```

=== ":fontawesome-brands-rust: Rust"

    ```rust
    use deepbiop::fastq;
    ```
