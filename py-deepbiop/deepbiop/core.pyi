# This file is automatically generated by pyo3_stub_gen
# ruff: noqa: E501, F401

import typing

def generate_kmers(base:str,k:int) -> list[str]:
    r"""
    Generate all possible k-mers from a set of base characters.

    This function takes a string of base characters and a k-mer length,
    and generates all possible k-mer combinations of that length.

    # Arguments

    * `base` - A string containing the base characters to use (e.g. "ATCG")
    * `k` - The length of k-mers to generate

    # Returns

    A vector containing all possible k-mer combinations as strings
    """

def generate_kmers_table(base:str,k:int) -> dict[list[int], int]:
    r"""
    Generate a lookup table mapping k-mers to unique IDs.

    This function takes a string of base characters and a k-mer length,
    and generates a HashMap mapping each possible k-mer to a unique integer ID.

    # Arguments

    * `base` - A string containing the base characters to use (e.g. "ATCG")
    * `k` - The length of k-mers to generate

    # Returns

    A HashMap mapping k-mer byte sequences to integer IDs
    """

def kmers_to_seq(kmers:typing.Sequence[str]) -> str:
    r"""
    Convert k-mers back into a DNA sequence.

    This function takes a vector of k-mers and reconstructs the original DNA sequence.
    The k-mers are assumed to be in order and overlapping.

    # Arguments

    * `kmers` - A vector of k-mers as `String`s

    # Returns

    The reconstructed DNA sequence as a `String`, wrapped in a `Result`
    """

def normalize_seq(seq:str,iupac:bool) -> str:
    r"""
    Normalize a DNA sequence by converting any non-standard nucleotides to standard ones.

    This function takes a DNA sequence as a `String` and a boolean flag `iupac` indicating whether to normalize using IUPAC ambiguity codes.
    It returns a normalized DNA sequence as a `String`.

    # Arguments

    * `seq` - A DNA sequence as a `String`.
    * `iupac` - A boolean flag indicating whether to normalize using IUPAC ambiguity codes.

    # Returns

    A normalized DNA sequence as a `String`.
    """

def reverse_complement(seq:str) -> str:
    r"""
    Generate the reverse complement of a DNA sequence.

    This function takes a DNA sequence as a `String` and returns its reverse complement.
    The reverse complement is generated by reversing the sequence and replacing each nucleotide
    with its complement (A<->T, C<->G).

    # Arguments

    * `seq` - A DNA sequence as a `String`

    # Returns

    The reverse complement sequence as a `String`

    # Example

    ```
    use deepbiop_core::seq::reverse_complement;

    let seq = String::from("ATCG");
    let rev_comp = reverse_complement(seq);
    assert_eq!(rev_comp, "CGAT");
    ```
    """

def seq_to_kmers(seq:str,k:int,overlap:bool) -> list[str]:
    r"""
    Convert a DNA sequence into k-mers.

    This function takes a DNA sequence and splits it into k-mers of specified length.
    The sequence is first normalized to handle non-standard nucleotides.

    # Arguments

    * `seq` - A DNA sequence as a `String`
    * `k` - The length of each k-mer
    * `overlap` - Whether to generate overlapping k-mers

    # Returns

    A vector of k-mers as `String`s
    """

