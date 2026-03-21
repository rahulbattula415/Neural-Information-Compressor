import time
import zstandard as zstd
from nic.pipeline import compress, load_tokenizer, load_model, uniform_probs, entropy_encode
import numpy as np

print("Loading model and tokenizer...")
tok = load_tokenizer()
lm = load_model()
print("Ready.\n")

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

def report(label: str, original_bytes: int, compressed_bytes: int, elapsed: float):
    ratio = compressed_bytes / original_bytes
    print(f"  {label:<20} {compressed_bytes:>6} bytes  ({ratio:.3f}x)  {elapsed*1000:>8.1f}ms")


# Warmup
print("Warming up...")
_warmup_text = "warmup"
compress(_warmup_text, tok, lm)
compress_uniform(_warmup_text)
zstd.ZstdCompressor().compress(_warmup_text.encode())
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

    (gpt2_compressed, n_tokens), gpt2_time = time_it(compress, text, tok, lm)
    (uniform_compressed, _), uniform_time = time_it(compress_uniform, text)
    zstd_compressed, zstd_time = time_it(compress_zstd, text)

    print(f"[{name}] — {original} bytes raw, {n_tokens} tokens")
    report("zstd",        original, len(zstd_compressed),    zstd_time)
    report("nic uniform", original, len(uniform_compressed), uniform_time)
    report("nic gpt2",    original, len(gpt2_compressed),    gpt2_time)
    print()