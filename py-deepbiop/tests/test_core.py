"""Tests for the Rust `deepbiop.core` module."""

from deepbiop import core


def test_reverse_complement():
    assert core.reverse_complement("ACTGAACCGAGATCGAGTG") == "CACTCGATCTCGGTTCAGT"


def test_seq_to_kmers():
    assert core.seq_to_kmers("ATCGA", 3, overlap=True) == ["ATC", "TCG", "CGA"]
    assert core.seq_to_kmers("ATCGAT", 3, overlap=False) == ["ATC", "GAT"]
