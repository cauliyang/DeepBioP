# DeepBioP

[![crates](https://img.shields.io/crates/v/deepbiop.svg)](https://crates.io/crates/deepbiop)
[![pypi](https://img.shields.io/pypi/v/deepbiop.svg)](https://pypi.python.org/pypi/deepbiop)
[![cli](https://img.shields.io/crates/v/deepbiop?label=CLI)](https://crates.io/crates/deepbiop-cli)
[![license](https://img.shields.io/pypi/l/deepbiop.svg)](https://github.com/cauliyang/DeepBioP/blob/main/LICENSE)
[![pypi version](https://img.shields.io/pypi/pyversions/deepbiop.svg)](https://pypi.python.org/pypi/deepbiop)
[![Actions status](https://github.com/cauliyang/DeepBioP/workflows/CI/badge.svg)](https://github.com/cauliyang/DeepBioP/actions)

Deep Learning Processing Library for Biological Data

# Setup

## Python

install the latest deepbiop version with:

```bash
pip install deepbiop
```

## Rust

You can take latest release from `crates.io`, or if you want to use the latest features / performance improvements point to the `main` branch of this repo.

```bash
cargo add deepbiop --features fq
```

Each enabled feature can then be imported by its re-exported name, e.g.,

```rust
use deepbiop::fastq;
```

### CLI

```bash
cargo install deepbiop-cli
dbp -h
```

# Minimum Supported Rust Version (MSRV)

This project adheres to a Minimum Supported Rust Version (MSRV) policy.
The Minimum Supported Rust Version (MSRV) is 1.75.0.
We ensure that all code within the project is compatible with this version or newer to maintain stability and compatibility.

# Contribute 🤝

**Call for Participation: Deep Learning Processing Library for Biological Data**

We are excited to announce the launch of a new open-source project focused on developing a cutting-edge deep learning processing library specifically designed for biological data.
This project aims to empower researchers, data scientists, and developers to leverage the latest advancements in deep learning to analyze and interpret complex biological datasets.

**Project Overview:**

Biological data, such as genomic sequences, proteomics, and imaging data, presents unique challenges and opportunities for machine learning applications.
Our library seeks to provide a comprehensive suite of tools and algorithms that streamline the preprocessing, modeling, and analysis of biological data using deep learning techniques.

**Key Features:**

- **Data Preprocessing:** Efficient tools for cleaning, normalizing, and augmenting biological data.
- **Model Building:** Pre-built models and customizable architectures tailored for various types of biological data.
- **Visualization:** Advanced visualization tools to help interpret model predictions and insights.
- **Integration:** Seamless integration with popular bioinformatics tools and frameworks.
- **APIs:** Rust and Python APIs to facilitate easy integration with different deep learning frameworks, ensuring efficient operations across platforms.

**Who Should Participate?**

We invite participation from individuals and teams who are passionate about bioinformatics, deep learning, and open-source software development.
Whether you are a researcher, developer, or student, your contributions can help shape the future of biological data analysis.

**How to Get Involved:**

- **Developers:** Contribute code, fix bugs, and develop new features.
- **Researchers:** Share your domain expertise and help validate models.
- **Students:** Gain experience by working on real-world data science problems.
- **Community Members:** Provide feedback, report issues, and help grow the user community.

**Join Us:**

If you are interested in participating, please visit our GitHub repository at [Github](---) to explore the project and get started.

<!-- You can also join our community forum at [Forum Link] for discussions, updates, and collaboration opportunities. -->

**Contact Us:**

For more information or questions, feel free to contact us at [yangyang.li@norwestern.edu].
We look forward to your participation and contributions to this exciting project!

**Together, let's advance the field of biological data analysis with the power of deep learning!**
