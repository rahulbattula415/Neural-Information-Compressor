import numpy as np
import constriction
from transformers import GPT2TokenizerFast

VOCAB_SIZE = 50_257


# Dummy model - returns uniform distribution over full GPT-2 vocab
def uniform_probs() -> np.ndarray:
    return np.full(VOCAB_SIZE, 1.0 / VOCAB_SIZE, dtype=np.float64)


# entropy_encode - ANS-encodes a token sequence using probability distribution. 
# Returns a uint32 array.
def entropy_encode(tokens: list[int], probs: np.ndarray) -> np.ndarray:
    model = constriction.stream.model.Categorical(probs, perfect=False)
    coder = constriction.stream.stack.AnsCoder()
    coder.encode_reverse(np.array(tokens, dtype=np.int32), model)
    return coder.get_compressed()  # np.ndarray[uint32]

# entropy_decode - ANS-decode a uint32 array back to token IDs.
def entropy_decode(compressed: np.ndarray, probs: np.ndarray, n_tokens: int) -> list[int]:
    model = constriction.stream.model.Categorical(probs, perfect=False)
    coder = constriction.stream.stack.AnsCoder(compressed)
    return coder.decode(model, n_tokens).tolist()


# compress - Tokenize and entropy-encode text. Returns (compressed_bytes, n_tokens).
def compress(text: str) -> tuple[bytes, int]:
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    tokens = tok.encode(text)
    probs = uniform_probs()
    compressed = entropy_encode(tokens, probs)
    return compressed.tobytes(), len(tokens)

# decompress - Entropy-decode and detokenize back to the original text.
def decompress(data: bytes, n_tokens: int) -> str:
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    probs = uniform_probs()
    compressed = np.frombuffer(data, dtype=np.uint32)
    tokens = entropy_decode(compressed, probs, n_tokens)
    return tok.decode(tokens)


# Smoke test
if __name__ == "__main__":
    sample = "Hello, this is a test of the Neural Information Compressor pipeline."

    print(f"Input : {sample!r}")

    compressed_bytes, n_tokens = compress(sample)
    raw_bytes = len(sample.encode())
    print(f"Tokens          : {n_tokens}")
    print(f"Raw bytes       : {raw_bytes}")
    print(f"Compressed bytes: {len(compressed_bytes)}  (uniform probs -> no gain expected)")

    recovered = decompress(compressed_bytes, n_tokens)
    print(f"Output: {recovered!r}")

    assert recovered == sample, f"Round-trip FAILED\n  expected: {sample!r}\n  got:      {recovered!r}"
    print("Round-trip OK")
