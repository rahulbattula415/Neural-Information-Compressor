import time
import zstandard as zstd
from nic.pipeline import compress, load_tokenizer, load_model, uniform_probs, entropy_encode, gpt2_probs_sequence, gpt2_probs_cached
import numpy as np

print("Loading model and tokenizer...")
tok = load_tokenizer()
lm = load_model()
print("Ready.\n")

def compress_prefix(text: str) -> tuple[bytes, int]:
    """Old O(n²) prefix-call path — kept for benchmark comparison only."""
    tokens = tok.encode(text)
    n = len(tokens)
    probs_seq = [uniform_probs()]
    for i in range(1, n):
        probs_seq.append(gpt2_probs_sequence(tokens[:i], lm)[-1])
    probs_seq = np.vstack(probs_seq)
    compressed = entropy_encode(tokens, probs_seq)
    return compressed.tobytes(), n

def compress_uniform(text: str) -> tuple[bytes, int]:
    tokens = tok.encode(text)
    n = len(tokens)
    probs_seq = np.vstack([uniform_probs() for _ in range(n)])
    compressed = entropy_encode(tokens, probs_seq)
    return compressed.tobytes(), n

def compress_zstd(text: str) -> bytes:
    cctx = zstd.ZstdCompressor()
    return cctx.compress(text.encode())

def time_it(fn, *args) -> tuple:
    start = time.perf_counter()
    result = fn(*args)
    elapsed = time.perf_counter() - start
    return result, elapsed

def report(label: str, original_bytes: int, compressed_bytes: int, elapsed: float, n_tokens: int = None):
    ratio = compressed_bytes / original_bytes
    per_token = f"  ({elapsed*1000/n_tokens:.1f}ms/tok)" if n_tokens else ""
    print(f"  {label:<24} {compressed_bytes:>6} bytes  ({ratio:.3f}x)  {elapsed*1000:>8.1f}ms{per_token}")

# Warmup — run each path once to warm CUDA and zstd
print("Warming up...")
_w = "warmup"
compress(_w, tok, lm)
compress_prefix(_w)
compress_uniform(_w)
zstd.ZstdCompressor().compress(_w.encode())
print("Done.\n")

texts = {
    "short sentence": "Hello, this is a test of the Neural Information Compressor pipeline.",
    "repetitive":     "the the the the the the the the the the the the the the the the",
    "natural prose":  (
        "The quick brown fox jumps over the lazy dog. "
        "Machine learning models can assign probabilities to sequences of tokens. "
        "Entropy coding exploits these probabilities to achieve compression."
    ),
}

for name, text in texts.items():
    original = len(text.encode())

    (gpt2_cached_compressed, n_tokens), cached_time   = time_it(compress,        text, tok, lm)
    (gpt2_prefix_compressed, _),        prefix_time   = time_it(compress_prefix,  text)
    (uniform_compressed, _),            uniform_time  = time_it(compress_uniform, text)
    zstd_compressed,                    zstd_time     = time_it(compress_zstd,    text)

    print(f"[{name}] — {original} bytes raw, {n_tokens} tokens")
    report("zstd",             original, len(zstd_compressed),         zstd_time)
    report("nic uniform",      original, len(uniform_compressed),      uniform_time,  n_tokens)
    report("nic gpt2 (prefix)",original, len(gpt2_prefix_compressed),  prefix_time,   n_tokens)
    report("nic gpt2 (cached)",original, len(gpt2_cached_compressed),  cached_time,   n_tokens)

    speedup = prefix_time / cached_time
    print(f"  speedup (cached vs prefix): {speedup:.2f}x  ({n_tokens} tokens)")
    print()

# Phase 4 scaling benchmark
import textwrap

long_texts = {
    64:  "The quick brown fox jumps over the lazy dog. " * 6,
    128: "The quick brown fox jumps over the lazy dog. " * 12,
    256: "The quick brown fox jumps over the lazy dog. " * 24,
    512: "The quick brown fox jumps over the lazy dog. " * 48,
}

print("=== Scaling benchmark (cached vs prefix) ===\n")
for target_tokens, text in long_texts.items():
    tokens = tok.encode(text)
    n = len(tokens)
    
    _, prefix_time = time_it(compress_prefix, text)
    _, cached_time = time_it(compress,        text, tok, lm)
    
    speedup = prefix_time / cached_time
    print(f"  ~{target_tokens:>3} tokens (actual: {n})  |  prefix: {prefix_time*1000:.0f}ms  cached: {cached_time*1000:.0f}ms  speedup: {speedup:.2f}x")