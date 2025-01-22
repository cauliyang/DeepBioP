# Command-Line Help for `deepbiop-cli`

This document contains the help content for the `deepbiop-cli` command-line program.

**Command Overview:**

* [`deepbiop-cli`↴](#deepbiop-cli)
* [`deepbiop-cli count-chimeric`↴](#deepbiop-cli-count-chimeric)
* [`deepbiop-cli bam-to-fq`↴](#deepbiop-cli-bam-to-fq)
* [`deepbiop-cli fq-to-fa`↴](#deepbiop-cli-fq-to-fa)
* [`deepbiop-cli fa-to-fq`↴](#deepbiop-cli-fa-to-fq)
* [`deepbiop-cli fa-to-parquet`↴](#deepbiop-cli-fa-to-parquet)
* [`deepbiop-cli fq-to-parquet`↴](#deepbiop-cli-fq-to-parquet)
* [`deepbiop-cli extract-fq`↴](#deepbiop-cli-extract-fq)
* [`deepbiop-cli extract-fa`↴](#deepbiop-cli-extract-fa)
* [`deepbiop-cli fqs-to-one`↴](#deepbiop-cli-fqs-to-one)
* [`deepbiop-cli fas-to-one`↴](#deepbiop-cli-fas-to-one)

## `deepbiop-cli`

CLI tool for Processing Biological Data.

**Usage:** `deepbiop-cli [OPTIONS] [COMMAND]`

###### **Subcommands:**

* `count-chimeric` — Count chimeric reads in a BAM file
* `bam-to-fq` — BAM to fastq conversion
* `fq-to-fa` — Fastq to fasta conversion
* `fa-to-fq` — Fasta to fastq conversion
* `fa-to-parquet` — Fasta to parquet conversion
* `fq-to-parquet` — Fastq to parquet conversion
* `extract-fq` — Extract fastq reads from a fastq file
* `extract-fa` — Extract fasta reads from a fasta file
* `fqs-to-one` — Multiple Fastqs to one Fastq conversion
* `fas-to-one` — Multiple Fastas to one Fasta conversion

###### **Options:**

* `--generate <GENERATOR>`

  Possible values: `bash`, `elvish`, `fish`, `powershell`, `zsh`

* `-v`, `--verbose` — Increase logging verbosity
* `-q`, `--quiet` — Decrease logging verbosity



## `deepbiop-cli count-chimeric`

Count chimeric reads in a BAM file

**Usage:** `deepbiop-cli count-chimeric [OPTIONS] [bam]...`

###### **Arguments:**

* `<bam>` — path to the bam file

###### **Options:**

* `-t`, `--threads <THREADS>` — threads number

  Default value: `2`



## `deepbiop-cli bam-to-fq`

BAM to fastq conversion

**Usage:** `deepbiop-cli bam-to-fq [OPTIONS] [bam]...`

###### **Arguments:**

* `<bam>` — path to the bam file

###### **Options:**

* `-t`, `--threads <THREADS>` — threads number

  Default value: `2`
* `-c`, `--compressed` — output bgzip compressed fastq file



## `deepbiop-cli fq-to-fa`

Fastq to fasta conversion

**Usage:** `deepbiop-cli fq-to-fa [OPTIONS] [fq]...`

###### **Arguments:**

* `<fq>` — path to the fq file

###### **Options:**

* `-t`, `--threads <THREADS>` — threads number

  Default value: `2`



## `deepbiop-cli fa-to-fq`

Fasta to fastq conversion

**Usage:** `deepbiop-cli fa-to-fq [OPTIONS] [fa]...`

###### **Arguments:**

* `<fa>` — path to the fa file

###### **Options:**

* `-t`, `--threads <THREADS>` — threads number

  Default value: `2`



## `deepbiop-cli fa-to-parquet`

Fasta to parquet conversion

**Usage:** `deepbiop-cli fa-to-parquet [OPTIONS] <fa>`

###### **Arguments:**

* `<fa>` — path to the fa file

###### **Options:**

* `--chunk` — if convert the fa file to parquet by chunk or not
* `--chunk-size <CHUNK_SIZE>` — chunk size

  Default value: `1000000`
* `--output <result>` — result path
* `-t`, `--threads <THREADS>` — threads number

  Default value: `2`



## `deepbiop-cli fq-to-parquet`

Fastq to parquet conversion

**Usage:** `deepbiop-cli fq-to-parquet [OPTIONS] <fq>`

###### **Arguments:**

* `<fq>` — path to the fq file

###### **Options:**

* `--chunk` — if convert the fq file to parquet by chunk or not
* `--chunk-size <CHUNK_SIZE>` — chunk size

  Default value: `1000000`
* `--output <output>` — result path
* `-t`, `--threads <THREADS>` — threads number

  Default value: `2`



## `deepbiop-cli extract-fq`

Extract fastq reads from a fastq file

**Usage:** `deepbiop-cli extract-fq [OPTIONS] <fq>`

###### **Arguments:**

* `<fq>` — path to the fq file

###### **Options:**

* `--reads <reads>` — Path to the selected reads
* `--number <number>` — The number of selected reads by random
* `--output <output>` — output bgzip compressed file
* `-t`, `--threads <THREADS>` — threads number

  Default value: `2`
* `-c`, `--compressed` — output bgzip compressed fastq file



## `deepbiop-cli extract-fa`

Extract fasta reads from a fasta file

**Usage:** `deepbiop-cli extract-fa [OPTIONS] <fa>`

###### **Arguments:**

* `<fa>` — path to the bam file

###### **Options:**

* `--reads <reads>` — Path to the selected reads
* `--number <number>` — The number of selected reads by random
* `--output <output>` — output bgzip compressed file
* `-t`, `--threads <THREADS>` — threads number

  Default value: `2`
* `-c`, `--compressed` — output bgzip compressed fasta file



## `deepbiop-cli fqs-to-one`

Multiple Fastqs to one Fastq conversion

**Usage:** `deepbiop-cli fqs-to-one [OPTIONS] --output <output> [fqs]...`

###### **Arguments:**

* `<fqs>` — path to the fq file

###### **Options:**

* `--output <output>` — output bgzip compressed file
* `-t`, `--threads <THREADS>`

  Default value: `2`



## `deepbiop-cli fas-to-one`

Multiple Fastas to one Fasta conversion

**Usage:** `deepbiop-cli fas-to-one [OPTIONS] --output <output> [fas]...`

###### **Arguments:**

* `<fas>` — path to the fa file

###### **Options:**

* `--output <output>` — output bgzip compressed file
* `-t`, `--threads <THREADS>`

  Default value: `2`



<hr/>

<small><i>
    This document was generated automatically by
    <a href="https://crates.io/crates/clap-markdown"><code>clap-markdown</code></a>.
</i></small>

