# Test Results

## Pre KV-Caching

### Short Sentence
**68 bytes raw · 14 tokens**

| Method      | Compressed | Ratio  | Time     |
|-------------|------------|--------|----------|
| zstd        | 68 bytes   | 1.000x | 0.1 ms   |
| nic uniform | 28 bytes   | 0.412x | 4.2 ms   |
| nic gpt2    | 16 bytes   | 0.235x | 378.5 ms |

---

### Repetitive
**63 bytes raw · 16 tokens**

| Method      | Compressed | Ratio  | Time     |
|-------------|------------|--------|----------|
| zstd        | 19 bytes   | 0.302x | 0.1 ms   |
| nic uniform | 32 bytes   | 0.508x | 4.2 ms   |
| nic gpt2    | 12 bytes   | 0.190x | 351.8 ms |

---

### Natural Prose
**185 bytes raw · 31 tokens**

| Method      | Compressed | Ratio  | Time     |
|-------------|------------|--------|----------|
| zstd        | 139 bytes  | 0.751x | 0.1 ms   |
| nic uniform | 64 bytes   | 0.346x | 8.4 ms   |
| nic gpt2    | 36 bytes   | 0.195x | 709.1 ms |

---

> **Note:** `nic gpt2` times include O(n²) GPT-2 forward passes (one per token prefix). KV-caching will reduce this to O(n).

---

## Post KV-Caching

### Short Sentence
**68 bytes raw · 14 tokens**

| Method            | Compressed | Ratio  | Time      | ms/tok |
|-------------------|------------|--------|-----------|--------|
| zstd              | 68 bytes   | 1.000x | 0.1 ms    | —      |
| nic uniform       | 28 bytes   | 0.412x | 5.6 ms    | 0.4    |
| nic gpt2 (prefix) | 16 bytes   | 0.235x | 327.1 ms  | 23.4   |
| nic gpt2 (cached) | 16 bytes   | 0.235x | 294.8 ms  | 21.1   |

**Speedup (cached vs prefix): 1.11x**

---

### Repetitive
**63 bytes raw · 16 tokens**

| Method            | Compressed | Ratio  | Time      | ms/tok |
|-------------------|------------|--------|-----------|--------|
| zstd              | 19 bytes   | 0.302x | 0.0 ms    | —      |
| nic uniform       | 32 bytes   | 0.508x | 3.2 ms    | 0.2    |
| nic gpt2 (prefix) | 12 bytes   | 0.190x | 294.5 ms  | 18.4   |
| nic gpt2 (cached) | 12 bytes   | 0.190x | 336.8 ms  | 21.1   |

**Speedup (cached vs prefix): 0.87x**

---

### Natural Prose
**185 bytes raw · 31 tokens**

| Method            | Compressed | Ratio  | Time      | ms/tok |
|-------------------|------------|--------|-----------|--------|
| zstd              | 139 bytes  | 0.751x | 0.1 ms    | —      |
| nic uniform       | 64 bytes   | 0.346x | 6.3 ms    | 0.2    |
| nic gpt2 (prefix) | 36 bytes   | 0.195x | 643.2 ms  | 20.7   |
| nic gpt2 (cached) | 36 bytes   | 0.195x | 657.3 ms  | 21.2   |

**Speedup (cached vs prefix): 0.98x**

---

### Scaling Benchmark (cached vs prefix)

| Target tokens | Actual tokens | Prefix   | Cached   | Speedup |
|---------------|---------------|----------|----------|---------|
| ~64           | 61            | 1469 ms  | 1325 ms  | 1.11x   |
| ~128          | 121           | 3335 ms  | 2549 ms  | 1.31x   |
| ~256          | 241           | 10825 ms | 5382 ms  | 2.01x   |
| ~512          | 481           | 55466 ms | 10301 ms | 5.38x   |

> **Note:** KV-cache speedup only becomes meaningful at longer sequences. At ~512 tokens the cached path is **5.38x faster**, consistent with H3 (>3x on long sequences). Short sequences see little benefit due to per-token overhead dominating.
