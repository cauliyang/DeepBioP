from deepbiop import core


def test_reverse_complement():
    result = core.reverse_complement("ACTGAACCGAGATCGAGTG")
    print(result)


def test_fq():
    result = core.seq_to_kmers("ATCGA", 3, overlap=True)
    expected = ["ATC", "TCG", "CGA"]
    assert result == expected
