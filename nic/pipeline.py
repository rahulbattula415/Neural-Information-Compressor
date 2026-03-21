import numpy as np
import constriction
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
import torch
from pathlib import Path

VOCAB_SIZE = 50_257
TOKENIZER_PATH = Path(__file__).parent / "tokenizer"
MODEL_PATH = Path(__file__).parent / "model"
PRECISION = 24

# Dummy model - returns uniform distribution over full GPT-2 vocab
def uniform_probs() -> np.ndarray:
    return np.full(VOCAB_SIZE, 1.0 / VOCAB_SIZE, dtype=np.float32)

# Gets tokenizer from json data from tokenizer directory
def load_tokenizer():
    return GPT2TokenizerFast.from_pretrained(TOKENIZER_PATH)

# Loads GPT-2 LM using json data from model directory
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return GPT2LMHeadModel.from_pretrained(MODEL_PATH).to(device).eval()
    
# Returns GPT-2 probability distribution
def gpt2_probs_sequence(tokens: list[int], model) -> np.ndarray:
    device = next(model.parameters()).device
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(input_ids).logits[0]
    probs = torch.softmax(logits, dim=-1).cpu().numpy().astype(np.float32)
    return probs

# Converts float probability array to integer frequency table
def probs_to_table(probs: np.ndarray) -> np.ndarray:
    table = np.floor(probs * (1 << PRECISION)).astype(np.int32)
    remainder = (1 << PRECISION) - table.sum()
    table[np.argmax(probs)] += remainder    
    return table

# entropy_encode - ANS-encodes a token sequence using per-token probability distributions.
# probs_seq: (n_tokens, vocab_size) float32 — probs_seq[i] is the distribution for tokens[i]
# ANS is a stack, so we push tokens in reverse order; decode will pop them in forward order.
def entropy_encode(tokens: list[int], probs_seq: np.ndarray) -> np.ndarray:
    coder = constriction.stream.stack.AnsCoder()
    for i in reversed(range(len(tokens))):
        model = constriction.stream.model.Categorical(probs_seq[i], perfect=False)
        coder.encode_reverse(np.array([tokens[i]], dtype=np.int32), model)
    return coder.get_compressed()  # np.ndarray[uint32]

# entropy_decode - ANS-decode a uint32 array back to token IDs.
# probs_seq must match exactly what was passed to entropy_encode.
def entropy_decode(compressed: np.ndarray, probs_seq: np.ndarray, n_tokens: int) -> list[int]:
    coder = constriction.stream.stack.AnsCoder(compressed)
    tokens = []
    for i in range(n_tokens):
        model = constriction.stream.model.Categorical(probs_seq[i], perfect=False)
        tokens.append(int(coder.decode(model, 1)[0]))
    return tokens


# compress - Tokenize and entropy-encode text. Returns (compressed_bytes, n_tokens).
# Probs are computed incrementally (one prefix call per token) to exactly match decompress.
def compress(text: str, tok=None, lm=None) -> tuple[bytes, int]:
    if tok is None:
        tok = load_tokenizer()
    if lm is None:
        lm = load_model()
    tokens = tok.encode(text)
    n = len(tokens)

    probs_seq = [uniform_probs()]
    for i in range(1, n):
        probs_seq.append(gpt2_probs_sequence(tokens[:i], lm)[-1])
    probs_seq = np.vstack(probs_seq)

    compressed = entropy_encode(tokens, probs_seq)
    return compressed.tobytes(), n

# decompress - Entropy-decode and detokenize back to the original text.
# Must reproduce the exact probs sequence used at encode time, so we decode autoregressively.
def decompress(data: bytes, n_tokens: int) -> str:
    tok = load_tokenizer()
    lm = load_model()
    compressed = np.frombuffer(data, dtype=np.uint32)
    coder = constriction.stream.stack.AnsCoder(compressed)

    tokens = []
    for i in range(n_tokens):
        if i == 0:
            probs = uniform_probs()
        else:
            probs = gpt2_probs_sequence(tokens, lm)[-1]  # P(·|tokens decoded so far)
        cat = constriction.stream.model.Categorical(probs, perfect=False)
        token = int(coder.decode(cat, 1)[0])
        tokens.append(token)

    return tok.decode(tokens)


# Smoke test
if __name__ == "__main__":
    sample = "Hello, this is a test of the Neural Information Compressor pipeline."

    print(f"Input : {sample!r}")

    compressed_bytes, n_tokens = compress(sample)
    raw_bytes = len(sample.encode())
    print(f"Tokens          : {n_tokens}")
    print(f"Raw bytes       : {raw_bytes}")
    print(f"Compressed bytes: {len(compressed_bytes)}")

    recovered = decompress(compressed_bytes, n_tokens)
    print(f"Output: {recovered!r}")

    assert recovered == sample, f"Round-trip FAILED\n  expected: {sample!r}\n  got:      {recovered!r}"
    print("Round-trip OK")
