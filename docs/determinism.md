# Determinism in NIC

## Why Determinism is Hard

NIC uses Asymmetric Numeral Systems (ANS) for entropy coding. ANS is extremely
sensitive to the probability model: if the encoder and decoder disagree on the
probability of even a single token, the entire bitstream from that point forward
is silently corrupted. There is no error message — you just get garbage output.

This means the probability distribution passed to the ANS coder must be
bit-identical between encode and decode, across:

- Multiple runs on the same machine
- Different machines
- Different operating systems
- CPU vs GPU inference (relevant from Phase 3 onwards)

## Sources of Non-Determinism

### 1. Tokenizer Drift
If the tokenizer is loaded from HuggingFace's servers, a future update to the
`gpt2` tokenizer files could change how text is tokenized. Different token
sequences mean different probability lookups, which corrupts the ANS stream.

### 2. Floating-Point Variance
Float32 arithmetic is not guaranteed to produce identical results across
hardware. GPU and CPU implementations of softmax can differ by ~1e-7, which
is enough to map to a different ANS probability bin and corrupt the stream.

### 3. Model Weight Drift
If model weights are loaded from a remote source without pinning to a specific
version, a repository update could silently change the probability outputs.

## Our Solutions

### Frozen Tokenizer
The GPT-2 tokenizer is saved locally to `nic/tokenizer/` using
`tokenizer.save_pretrained()` and always loaded from that path. It never
contacts the network at runtime.

### Fixed-Precision Probability Table (probs_to_table)
Before passing probabilities to the ANS coder, we convert them to an integer
frequency table that sums exactly to `2^PRECISION` (currently `2^24`):
```python
def probs_to_table(probs: np.ndarray) -> np.ndarray:
    table = np.floor(probs * (1 << PRECISION)).astype(np.int32)
    remainder = (1 << PRECISION) - table.sum()
    table[np.argmax(probs)] += remainder
    return table
```

Integer arithmetic is exact and hardware-independent. Any floating-point
variance in the model outputs is eliminated before it can affect the ANS coder.
This function is in place now and will be wired into real GPT-2 inference in
Phase 3.

### Model Weight Pinning (Phase 3)
When real GPT-2 inference is added, weights will be loaded with a pinned commit
SHA (`revision="<sha>"`) to prevent silent upstream changes.

## What We Tested

- Round-trip: compress → decompress returns original text exactly
- Repeated runs: same input produces bit-identical `.nic` output across 5 runs
- Raw byte stability: two compress calls return identical byte objects
- Frequency table: `probs_to_table` sums to exactly `2^24` with no zero entries
- CPU vs GPU: test fixture in place, skipped until Phase 3 adds GPU inference

## Remaining Risk

The current dummy model (uniform distribution) makes determinism trivial since
`np.full(..., dtype=np.float32)` is always identical. The real test comes in
Phase 3 when GPT-2 softmax outputs replace the uniform distribution. At that
point, `probs_to_table` becomes load-bearing and the CPU/GPU test must pass
with actual inference.