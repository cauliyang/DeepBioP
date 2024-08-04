from deepbiop import bam


def test_cigar():
    result = bam.left_right_soft_clip("10S10M4S")
    expected = (10, 4)
    assert result == expected
