# Command-Line Help for `deepbiop-cli`

This document contains the help content for the `deepbiop-cli` command-line program.

**Command Overview:**

- [Command-Line Help for `deepbiop-cli`](#command-line-help-for-deepbiop-cli)
  - [`deepbiop-cli`](#deepbiop-cli)
          - [**Subcommands:**](#subcommands)
          - [**Options:**](#options)
  - [`deepbiop-cli count-chimeric`](#deepbiop-cli-count-chimeric)
          - [**Arguments:**](#arguments)
          - [**Options:**](#options-1)
  - [`deepbiop-cli bam-to-fq`](#deepbiop-cli-bam-to-fq)
          - [**Arguments:**](#arguments-1)
          - [**Options:**](#options-2)
  - [`deepbiop-cli fq-to-fa`](#deepbiop-cli-fq-to-fa)
          - [**Arguments:**](#arguments-2)
          - [**Options:**](#options-3)
  - [`deepbiop-cli fa-to-fq`](#deepbiop-cli-fa-to-fq)
          - [**Arguments:**](#arguments-3)
          - [**Options:**](#options-4)
  - [`deepbiop-cli fx-to-parquet`](#deepbiop-cli-fx-to-parquet)
          - [**Arguments:**](#arguments-4)
          - [**Options:**](#options-5)
  - [`deepbiop-cli extract-fx`](#deepbiop-cli-extract-fx)
          - [**Arguments:**](#arguments-5)
          - [**Options:**](#options-6)
  - [`deepbiop-cli fxs-to-one`](#deepbiop-cli-fxs-to-one)
          - [**Arguments:**](#arguments-6)
          - [**Options:**](#options-7)
  - [`deepbiop-cli count-fx`](#deepbiop-cli-count-fx)
          - [**Arguments:**](#arguments-7)
          - [**Options:**](#options-8)

## `deepbiop-cli`

CLI tool for Processing Biological Data.

**Usage:** `deepbiop-cli [OPTIONS] [COMMAND]`

###### **Subcommands:**

- `count-chimeric` — Count chimeric reads in a BAM file
- `bam-to-fq` — BAM to fastq conversion
- `fq-to-fa` — Fastq to fasta conversion
- `fa-to-fq` — Fasta to fastq conversion
- `fx-to-parquet` — Fastx to parquet conversion
- `extract-fx` — Extract reads from a fastx file
- `fxs-to-one` — Multiple Fastxs to one Fastx conversion
- `count-fx` — Profile sequences in a fasta file

###### **Options:**

- `--generate <GENERATOR>`

  Possible values: `bash`, `elvish`, `fish`, `powershell`, `zsh`

- `-v`, `--verbose` — Increase logging verbosity
- `-q`, `--quiet` — Decrease logging verbosity

## `deepbiop-cli count-chimeric`

Count chimeric reads in a BAM file

**Usage:** `deepbiop-cli count-chimeric [OPTIONS] [bam]...`

###### **Arguments:**

- `<bam>` — path to the bam file

###### **Options:**

- `-t`, `--threads <THREADS>` — threads number

  Default value: `2`

## `deepbiop-cli bam-to-fq`

BAM to fastq conversion

**Usage:** `deepbiop-cli bam-to-fq [OPTIONS] [bam]...`

###### **Arguments:**

- `<bam>` — path to the bam file

###### **Options:**

- `-t`, `--threads <THREADS>` — threads number

  Default value: `2`

- `-c`, `--compressed` — output bgzip compressed fastq file

## `deepbiop-cli fq-to-fa`

Fastq to fasta conversion

**Usage:** `deepbiop-cli fq-to-fa [OPTIONS] [fq]...`

###### **Arguments:**

- `<fq>` — path to the fq file

###### **Options:**

- `-t`, `--threads <THREADS>` — threads number

  Default value: `2`

## `deepbiop-cli fa-to-fq`

Fasta to fastq conversion

**Usage:** `deepbiop-cli fa-to-fq [OPTIONS] [fa]...`

###### **Arguments:**

- `<fa>` — path to the fa file

###### **Options:**

- `-t`, `--threads <THREADS>` — threads number

  Default value: `2`

## `deepbiop-cli fx-to-parquet`

Fastx to parquet conversion

**Usage:** `deepbiop-cli fx-to-parquet [OPTIONS] <fx>`

###### **Arguments:**

- `<fx>` — path to the fx file

###### **Options:**

- `--chunk` — if convert the fa file to parquet by chunk or not
- `--chunk-size <CHUNK_SIZE>` — chunk size

  Default value: `1000000`

- `--output <result>` — result path
- `-t`, `--threads <THREADS>` — threads number

  Default value: `2`

## `deepbiop-cli extract-fx`

Extract reads from a fastx file

**Usage:** `deepbiop-cli extract-fx [OPTIONS] <fx>`

###### **Arguments:**

- `<fx>` — path to the fastx file

###### **Options:**

- `--reads <reads>` — Path to the selected reads
- `--number <number>` — The number of selected reads by random
- `--output <output>` — output bgzip compressed file
- `-t`, `--threads <THREADS>` — threads number

  Default value: `2`

- `-c`, `--compressed` — output bgzip compressed fasta file

## `deepbiop-cli fxs-to-one`

Multiple Fastxs to one Fastx conversion

**Usage:** `deepbiop-cli fxs-to-one [OPTIONS] --output <output> [fxs]...`

###### **Arguments:**

- `<fxs>` — path to the fx file

###### **Options:**

- `--output <output>` — output bgzip compressed file
- `-t`, `--threads <THREADS>`

  Default value: `2`

## `deepbiop-cli count-fx`

Profile sequences in a fasta file

**Usage:** `deepbiop-cli count-fx [OPTIONS] <fx>`

###### **Arguments:**

- `<fx>` — path to the fastx file

###### **Options:**

- `--export` — if export the result
- `-t`, `--threads <THREADS>` — threads number

  Default value: `2`

<hr/>

<small><i>
This document was generated automatically by
<a href="https://crates.io/crates/clap-markdown"><code>clap-markdown</code></a>.
</i></small>
