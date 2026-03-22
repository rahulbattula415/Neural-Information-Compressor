import hashlib
import numpy as np
import pytest
from nic.pipeline import compress, decompress, probs_to_table, uniform_probs, gpt2_probs_cached, gpt2_probs_sequence, load_tokenizer, load_model, PRECISION

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

def test_gpt2_round_trip_deterministic():
    text = "Hello, this is a test of the Neural Information Compressor pipeline."

    compressed1, n_tokens1 = compress(text)
    compressed2, n_tokens2 = compress(text)

    assert n_tokens1 == n_tokens2
    assert compressed1 == compressed2, "Compressed bytes differ between runs"

    recovered = decompress(compressed1, n_tokens1)
    assert recovered == text, f"Round-trip failed: got {recovered!r}"

def test_kv_cache_compress_decompress_match():
    """
    Both sides using the cached path must produce identical tables.
    They don't need to match prefix calls — just each other.
    """
    tok = load_tokenizer()
    lm = load_model()
    tokens = tok.encode(SAMPLE)

    cached_probs_a = gpt2_probs_cached(tokens, lm)
    cached_probs_b = gpt2_probs_cached(tokens, lm)

    for i in range(len(tokens)):
        table_a = probs_to_table(cached_probs_a[i])
        table_b = probs_to_table(cached_probs_b[i])
        if not np.array_equal(table_a, table_b):
            pytest.fail(f"Cached path is not self-consistent at token {i}")

def test_kv_cache_matches_prefix_calls():
    """
    KV-cached probability tables must be identical to prefix-call tables
    after quantization. Raw float differences up to ~1e-7 are acceptable
    if probs_to_table absorbs them — this test checks at the boundary that
    actually matters for lossless recovery.
    """
    from nic.pipeline import load_tokenizer, load_model

    tok = load_tokenizer()
    lm = load_model()
    tokens = tok.encode(SAMPLE)

    prefix_probs = []
    for i in range(len(tokens)):
        if i == 0:
            p = uniform_probs()
        else:
            p = gpt2_probs_sequence(tokens[:i], lm)[-1]
        prefix_probs.append(p)
    prefix_probs = np.array(prefix_probs)

    cached_probs = gpt2_probs_cached(tokens, lm)

    assert prefix_probs.shape == cached_probs.shape, (
        f"Shape mismatch: prefix={prefix_probs.shape}, cached={cached_probs.shape}"
    )

    for i in range(len(tokens)):
        prefix_table = probs_to_table(prefix_probs[i])
        cached_table = probs_to_table(cached_probs[i])
        if not np.array_equal(prefix_table, cached_table):
            max_prob_diff = np.abs(prefix_probs[i] - cached_probs[i]).max()
            pytest.fail(
                f"Quantized table mismatch at token {i} ('{tok.decode([tokens[i]])}').\n"
                f"Max float diff before quantization: {max_prob_diff:.2e}\n"
                f"Caching produces different entropy coding tables — lossless recovery broken."
            )