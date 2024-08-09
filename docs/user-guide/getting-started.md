# Getting started

This chapter is here to help you get started with DeepBiop.
It covers all the fundamental features and functionalities of the library, making it easy for new users to familiarise themselves with the basics from initial installation and setup to core functionalities.
If you're already an advanced user or familiar with Dataframes, feel free to skip ahead to the [next chapter about installation options](installation.md).

## Installing deepbiop

=== ":fontawesome-brands-python: Python"

    ```bash
    pip install deepbiop
    ```

=== ":fontawesome-brands-rust: Rust"

    ```shell
    cargo add deepbiop -F fastq

    # Or Cargo.toml
    [dependencies]
    deepbiop = { version = "x", features = ["fastq", ...]}
    ```

<!-- In the example below you see that we select `col('*')`. The asterisk stands for all columns. -->

<!-- {{code_block('user-guide/getting-started/expressions','select',\['select'\])}} -->

<!-- ```python exec="on" result="text" session="getting-started/expressions" -->

<!-- --8<-- "python/user-guide/getting-started/expressions.py:setup" -->

<!-- print( -->

<!--     --8<-- "python/user-guide/getting-started/expressions.py:select" -->

<!-- ) -->

<!-- ``` -->
