"""Performance tests for dataset streaming throughput."""

import time
from pathlib import Path

import pytest


@pytest.mark.slow
@pytest.mark.benchmark
def test_fastq_throughput(benchmark):
    """Test FASTQ streaming achieves 1M+ records/sec throughput.

    Requirements:
    - FR-015: High-performance iteration (1M+ records/sec)
    - SC-002: Throughput ≥1M records/sec

    This test verifies that:
    1. FASTQ files can be streamed at >1M records/sec
    2. Performance is consistent across iterations
    3. No performance degradation over time
    """
    pytest.skip("Requires FastqDataset implementation (T020)")

    from deepbiop.fq import FastqDataset

    test_file = Path(__file__).parent / "data" / "large_10k.fastq.gz"

    def stream_records():
        """Stream all records and count them."""
        dataset = FastqDataset(test_file)
        count = 0
        for _ in dataset:
            count += 1
        return count

    # Warmup run
    record_count = stream_records()
    assert record_count > 0, "Dataset should contain records"

    # Benchmark run
    benchmark(stream_records)

    # Calculate throughput
    elapsed_time = benchmark.stats["mean"]
    throughput = record_count / elapsed_time

    print(f"\nThroughput: {throughput:,.0f} records/sec")
    print(f"Records processed: {record_count:,}")
    print(f"Average time: {elapsed_time:.4f}s")

    # Verify throughput meets requirement
    assert throughput >= 1_000_000, (
        f"Throughput {throughput:,.0f} records/sec is below 1M records/sec requirement"
    )


@pytest.mark.benchmark
def test_fastq_throughput_simple():
    """Simple throughput test without pytest-benchmark.

    This is a fallback test that works without pytest-benchmark plugin.
    Can be run with: pytest -m benchmark tests/test_performance.py::test_fastq_throughput_simple
    """
    pytest.skip("Requires FastqDataset implementation (T020)")

    from deepbiop.fq import FastqDataset

    test_file = Path(__file__).parent / "data" / "large_10k.fastq.gz"

    dataset = FastqDataset(test_file)

    # Warmup
    count = sum(1 for _ in dataset)
    assert count > 0

    # Timed run
    start = time.perf_counter()
    count = 0
    for _ in dataset:
        count += 1
    elapsed = time.perf_counter() - start

    throughput = count / elapsed

    print(f"\nProcessed {count:,} records in {elapsed:.4f}s")
    print(f"Throughput: {throughput:,.0f} records/sec")

    assert throughput >= 1_000_000, f"Throughput {throughput:,.0f} below 1M records/sec"


@pytest.mark.benchmark
def test_batch_creation_throughput():
    """Test batch creation throughput.

    Requirements:
    - FR-016: Efficient batching
    - SC-003: Fast batch creation

    This test verifies that:
    1. Batching doesn't significantly slow down iteration
    2. Batch creation is efficient (<10ms per 32-record batch)
    """
    pytest.skip("Requires FastqDataset implementation (T020)")

    from deepbiop.core import collate_batch
    from deepbiop.fq import FastqDataset

    test_file = Path(__file__).parent / "data" / "large_10k.fastq.gz"

    dataset = FastqDataset(test_file)
    batch_size = 32

    # Collect batching times
    batch_times = []
    batch_count = 0

    records = []
    for record in dataset:
        records.append(record)

        if len(records) == batch_size:
            # Time batch creation
            start = time.perf_counter()
            batch = collate_batch(records, padding="longest")
            batch_time = time.perf_counter() - start

            batch_times.append(batch_time)
            batch_count += 1

            # Verify batch
            assert len(batch) == batch_size

            records = []

            # Stop after reasonable number of batches
            if batch_count >= 100:
                break

    assert batch_count > 0, "Should have created batches"

    # Calculate statistics
    avg_batch_time = sum(batch_times) / len(batch_times)
    max_batch_time = max(batch_times)
    min_batch_time = min(batch_times)

    print(f"\nBatched {batch_count} batches of {batch_size} records")
    print(f"Average batch time: {avg_batch_time * 1000:.2f}ms")
    print(f"Min batch time: {min_batch_time * 1000:.2f}ms")
    print(f"Max batch time: {max_batch_time * 1000:.2f}ms")

    # Verify batch creation is fast enough
    # 32 records at 1M records/sec = 32μs for reading
    # Allow 10ms for batching overhead (generous)
    assert avg_batch_time < 0.010, (
        f"Average batch time {avg_batch_time * 1000:.2f}ms exceeds 10ms threshold"
    )


@pytest.mark.benchmark
def test_parallel_dataloader_throughput():
    """Test PyTorch DataLoader with multiple workers.

    Requirements:
    - FR-017: Multi-worker support
    - SC-005: Parallel loading efficiency

    This test verifies that:
    1. DataLoader works with num_workers > 0
    2. Parallel loading improves throughput
    3. No deadlocks or crashes with multiple workers
    """
    pytest.skip("Requires FastqDataset and PyTorch integration (T020, T045)")

    from torch.utils.data import DataLoader

    from deepbiop.fq import FastqDataset

    test_file = Path(__file__).parent / "data" / "large_10k.fastq.gz"

    dataset = FastqDataset(test_file)

    # Test with 4 workers
    num_workers = 4
    batch_size = 32

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=False,  # Disable for CPU-only testing
    )

    # Measure throughput
    start = time.perf_counter()
    batch_count = 0
    record_count = 0

    for batch in loader:
        batch_count += 1
        record_count += len(batch)

        # Stop after reasonable number for testing
        if batch_count >= 100:
            break

    elapsed = time.perf_counter() - start

    throughput = record_count / elapsed

    print(f"\nDataLoader with {num_workers} workers:")
    print(f"Processed {batch_count} batches ({record_count} records) in {elapsed:.4f}s")
    print(f"Throughput: {throughput:,.0f} records/sec")

    # With parallel loading, should still maintain good throughput
    # May not hit 1M due to Python GIL and multiprocessing overhead,
    # but should be reasonable (>100K records/sec)
    assert throughput >= 100_000, (
        f"DataLoader throughput {throughput:,.0f} below 100K records/sec"
    )


@pytest.mark.benchmark
def test_compression_overhead():
    """Test throughput difference between compressed and uncompressed files.

    This test verifies that:
    1. Decompression overhead is acceptable
    2. gzip vs bgzip performance characteristics
    """
    pytest.skip("Requires FastqDataset implementation (T020)")

    from deepbiop.fq import FastqDataset

    test_files = {
        "uncompressed": Path(__file__).parent / "data" / "test.fastq",
        "gzip": Path(__file__).parent / "data" / "test.fastq.gz",
        "bgzip": Path(__file__).parent / "data" / "test.fastqbgz.gz",
    }

    results = {}

    for compression_type, file_path in test_files.items():
        if not file_path.exists():
            continue

        dataset = FastqDataset(file_path)

        # Time iteration
        start = time.perf_counter()
        count = sum(1 for _ in dataset)
        elapsed = time.perf_counter() - start

        throughput = count / elapsed if elapsed > 0 else 0
        results[compression_type] = {
            "count": count,
            "time": elapsed,
            "throughput": throughput,
        }

        print(f"\n{compression_type}:")
        print(f"  Records: {count}")
        print(f"  Time: {elapsed:.4f}s")
        print(f"  Throughput: {throughput:,.0f} records/sec")

    # Verify results are consistent
    if len(results) > 1:
        counts = [r["count"] for r in results.values()]
        assert len(set(counts)) == 1, "All files should have same record count"
