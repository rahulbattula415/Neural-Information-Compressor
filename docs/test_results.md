# Test Results

## Pre KV-Caching

### Short Sentence
**68 bytes raw · 14 tokens**

| Method      | Compressed | Ratio  | Time     |
|-------------|------------|--------|----------|
| zstd        | 68 bytes   | 1.000x | 93.1 ms  |
| nic uniform | 28 bytes   | 0.412x | 3.4 ms   |
| nic gpt2    | 16 bytes   | 0.235x | 888.6 ms |

---

### Repetitive
**63 bytes raw · 16 tokens**

| Method      | Compressed | Ratio  | Time     |
|-------------|------------|--------|----------|
| zstd        | 19 bytes   | 0.302x | 0.1 ms   |
| nic uniform | 32 bytes   | 0.508x | 3.8 ms   |
| nic gpt2    | 12 bytes   | 0.190x | 327.3 ms |

---

### Natural Prose
**185 bytes raw · 31 tokens**

| Method      | Compressed | Ratio  | Time     |
|-------------|------------|--------|----------|
| zstd        | 139 bytes  | 0.751x | 0.1 ms   |
| nic uniform | 64 bytes   | 0.346x | 8.1 ms   |
| nic gpt2    | 36 bytes   | 0.195x | 690.8 ms |

---

> **Note:** `nic gpt2` times include O(n²) GPT-2 forward passes (one per token prefix). KV-caching will reduce this to O(n).
