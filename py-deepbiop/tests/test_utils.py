from deepbiop import utils


def test_hight_targets():
    result = utils.highlight_targets("ACTGAACCGAGATCGAGTG", [(0, 3), (7, 10)])
    print(result)


# def test_segment():
#     print(utils.__all__)

#     segment = utils.GenomicInterval("chr10", 300, 5000)
#     segment2 = utils.GenomicInterval("chr10", 300, 5000)

#     assert segment.overlap(segment2)

#     print(segment)


def test_reverse_complement():
    result = utils.reverse_complement("ACTGAACCGAGATCGAGTG")
    print(result)
