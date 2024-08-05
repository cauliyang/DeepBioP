from deepbiop import utils


def test_hight_targets():
    result = utils.highlight_targets("ACTGAACCGAGATCGAGTG", [(0, 3), (7, 10)])
    print(result)
