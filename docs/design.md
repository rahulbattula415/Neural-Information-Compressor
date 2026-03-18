# Overview: Neural Information Compressor (NIC)

This project aims to study the role that LLMs could play in the compression of different text based files.

---

## Model: GPT-2

For this project, I chose to use the GPT-2 model. This was decided for reproducibility, local inference, and deterministic behavior rather than maximum compression performance. It is also valuable as an open source model that can run locally with no API costs, network dependency, or rate limits, allowing for vast experimentation.

---

## Context Strategy: Sliding Window

The maximum amount of context tokens allowed within GPT-2 is 1024. In order to allow the most accurate predictions, we require the maximum amount of context, meaning we use the full 1024 context tokens in a sliding window fashion. Although this seems like a large amount of data to keep in memory, the planned implementation of KV caching will improve the runtime.

---

## Pipeline Structure

The input for the NIC pipeline during compression will be a text based file, including `.txt`, `.json`, `.log`, and source code files. We will split this file into chunks. We will be testing multiple different chunk sizes (128, 256, 512, 1024) in order to see what results in efficient compression. If the chunk has high entropy, meaning there is more randomness in the chunk, we will use zstd to compress it. If the chunk has low entropy, we will use the LLM. From here, we will use entropy coding. This is an implementation of the idea that if there is a higher probability that something will occur, then we require less bits to represent it. Entropy coding exploits this idea to create a compact bitstream. The output will be a `.nic` file, containing the compressed bitstream as well as metadata in the header to reconstruct the file during decompression.

Decompression follows a similar pipeline, except backwards. The input will be a `.nic` file. The header of this file will store magic bytes, version, model_id, tokenizer_id, chunk_size, bitstream offset, and codec_per_chunk. For each chunk, we will check the codec_per_chunk field, and hand it to zstd if zstd compressed. If the LLM compressed, we will look at all the tokens we've recovered so far for context. Then, we will ask GPT-2 what's likely to come next, giving us probabilities which we feed into the entropy decoder. This reads a few bits from the bitstream and figures out what the token was, then adds that to the list.

---

## Hypotheses

**H1** - The hybrid compression pipeline of NIC will be more efficient in compression than just using zstd for structured text files, specifically `.json` files. These files have a more predictable structure, which should be interpreted by the GPT-2 model.

**H2** - The residual entropy after zstd data compression should be lower on natural language when compared to more structured text files, like `.json` files. This would mean that zstd extracts more compressible information from natural language, while some structure will remain within the structured text files. This would explain why the hybrid compression pipeline is compressing more with structured text files than natural language text files.

**H3** - The KV-cache implementation should improve the amount of data compressed per unit of time by more than 3x on long sequences.
