from deepbiop import utils


def test_hight_targets():
    result = utils.highlight_targets("ACTGAACCGAGATCGAGTG", [(0, 3), (7, 10)])
    print(result)


def test_segment():
    from deepbiop.utils import Segment

    segment = Segment("chr10", 300, 5000)
    segment2 = utils.Segment("chr10", 300, 5000)

    assert segment.overlap(segment2)

    print(segment)
