from deepbiop import fq


def test_fq():
    result = fq.seq_to_kmers("ATCGA", 3, overlap=True)
    expected = ["ATC", "TCG", "CGA"]
    assert result == expected
