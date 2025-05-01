from deepbiop import fq


def test_read_fq():
    pass


def test_fq_dataset():
    dataset = fq.FastqDataset("tests/data/test.fastq", 1)
    # flatten fq1
    fq1 = [item for sublist in dataset for item in sublist]
    assert len(fq1) == 25
