import hashlib
import numpy as np
import pytest
from nic.pipeline import compress, decompress, probs_to_table, uniform_probs

SAMPLE = "Hello, this is a test of the Neural Information Compressor pipeline."


def hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_round_trip():
    """Compress then decompress returns original text."""
    compressed, n_tokens = compress(SAMPLE)
    recovered = decompress(compressed, n_tokens)
    assert recovered == SAMPLE


def test_identical_output_repeated_runs():
    """Same input compressed multiple times produces bit-identical output."""
    results = [hash_bytes(compress(SAMPLE)[0]) for _ in range(5)]
    assert len(set(results)) == 1, "Compression is not deterministic across runs"


def test_compressed_bytes_are_stable():
    """Compress twice and confirm raw bytes are identical, not just the hash."""
    compressed_a, n_a = compress(SAMPLE)
    compressed_b, n_b = compress(SAMPLE)
    assert compressed_a == compressed_b
    assert n_a == n_b


def test_probs_to_table_sums_to_precision():
    """probs_to_table output must sum exactly to 2^PRECISION."""
    from nic.pipeline import PRECISION
    probs = uniform_probs()
    table = probs_to_table(probs)
    assert table.sum() == (1 << PRECISION), "Frequency table does not sum to 2^PRECISION"


def test_probs_to_table_no_zeros():
    """No token should have zero probability (would break ANS)."""
    probs = uniform_probs()
    table = probs_to_table(probs)
    assert np.all(table > 0), "Some tokens have zero probability in frequency table"


@pytest.mark.skipif(
    not __import__("torch").cuda.is_available(),
    reason="CUDA not available"
)
def test_cpu_gpu_identical_output():
    """CPU and GPU compression must produce identical output."""
    import torch
    # With dummy uniform model there's no GPU path yet,
    # but this fixture is here for Phase 3 when real inference is added.
    compressed_cpu, n_cpu = compress(SAMPLE)
    # Phase 3: add GPU inference path here and compare
    compressed_gpu, n_gpu = compress(SAMPLE)
    assert compressed_cpu == compressed_gpu
    assert n_cpu == n_gpu