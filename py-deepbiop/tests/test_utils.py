import deepbiop


def test_hight_targets():
    result = deepbiop.utils.highlight_targets("ACTGAACCGAGATCGAGTG", [(0, 3), (7, 10)])
    print(result)
