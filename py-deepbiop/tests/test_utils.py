from deepbiop import utils


def test_hight_targets():
    result = utils.highlight_targets("ACTGAACCGAGATCGAGTG", [(0, 3), (7, 10)])
    print(result)


def test_dectect_compressed_file():
    file = "./tests/data/test.fastq"
    result = utils.check_compressed_type(file)
    assert result == utils.CompressedType.Uncompress

    file = "./tests/data/test.fastq.gz"
    result = utils.check_compressed_type(file)
    assert result == utils.CompressedType.Gzip

    file = "./tests/data/test.fastqbgz.gz"
    result = utils.check_compressed_type(file)
    assert result == utils.CompressedType.Bgzip
