import argparse
from pathlib import Path
from nic.pipeline import compress, decompress
from nic.format import NicHeader, HEADER_FIXED_SIZE

VERSION = 1
MODEL_ID = "uniform"
TOKENIZER_ID = "gpt2"
CODEC_PLACEHOLDER = b"\x00"  # Phase 1 placeholder; real routing (zstd=0x00, llm=0x01) added in Phase 5


def cmd_compress(args):
    src = Path(args.file)
    text = src.read_text(encoding="utf-8")
    compressed_bytes, n_tokens = compress(text)

    header = NicHeader(
        version=VERSION,
        model_id=MODEL_ID,
        tokenizer_id=TOKENIZER_ID,
        chunk_size=n_tokens,  # TODO Phase 5: chunk_size should be the fixed split size (128/256/512); n_tokens here is the whole-file token count since we treat the file as one chunk
        num_chunks=1,
        codec_per_chunk=CODEC_PLACEHOLDER,
        bitstream_offset=HEADER_FIXED_SIZE + 1,  # fixed header + 1 codec byte
    )
    dest = src.with_suffix(".nic")
    dest.write_bytes(header.pack() + compressed_bytes)
    print(f"Compressed {src} -> {dest}  ({len(text.encode())} bytes -> {dest.stat().st_size} bytes, {n_tokens} tokens)")


def cmd_decompress(args):
    src = Path(args.file)
    raw = src.read_bytes()

    header = NicHeader.unpack(raw)
    compressed_bytes = raw[header.bitstream_offset:]

    text = decompress(compressed_bytes, header.chunk_size)
    dest = src.with_suffix(".txt")
    dest.write_text(text, encoding="utf-8")
    print(f"Decompressed {src} -> {dest}")


def main():
    parser = argparse.ArgumentParser(prog="nic")
    sub = parser.add_subparsers(dest="command", required=True)

    p_compress = sub.add_parser("compress", help="Compress a text file to .nic")
    p_compress.add_argument("file", help="Input text file")
    p_compress.set_defaults(func=cmd_compress)

    p_decompress = sub.add_parser("decompress", help="Decompress a .nic file")
    p_decompress.add_argument("file", help="Input .nic file")
    p_decompress.set_defaults(func=cmd_decompress)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
