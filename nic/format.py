import struct, dataclasses

MAGIC = b"NIC\x00" # Sanity check - use this to know that our file is .nic
HEADER_FMT = ">4s H 32s 32s I I Q" # contains the fixed size portion of our metadata
HEADER_FIXED_SIZE = struct.calcsize(HEADER_FMT) # Size of header fmt

# NicHeader - contains all needed information of a .nic file needed to revert the 
# compressed file back to normal.
@dataclasses.dataclass
class NicHeader:
    version: int
    model_id: str
    tokenizer_id: str
    chunk_size: int
    num_chunks: int
    codec_per_chunk: bytes
    bitstream_offset: int

    # pack - uses metadata of NicHeader object and uses struct.pack() method to 
    # convert to raw bytes.
    def pack(self) -> bytes:
        fixed = struct.pack(
            HEADER_FMT, 
            MAGIC,
            self.version,
            self.model_id.encode().ljust(32, b"\x00")[:32],
            self.tokenizer_id.encode().ljust(32, b"\x00")[:32],
            self.chunk_size,
            self.num_chunks,
            self.bitstream_offset
        )
        return fixed + self.codec_per_chunk

    # unpack - takes raw bytes from .nic file, and converts it back to an
    # NicHeader object.
    @classmethod
    def unpack(cls, data: bytes) -> "NicHeader":
        fixed = data[:HEADER_FIXED_SIZE]
        magic, version, model_id, tokenizer_id, chunk_size, num_chunks, bitstream_offset = struct.unpack(HEADER_FMT, fixed)

        assert magic == MAGIC, f"Invalid magic bytes: {magic!r}"

        codec_per_chunk = data[HEADER_FIXED_SIZE: HEADER_FIXED_SIZE + num_chunks]

        return cls(
            version=version,
            model_id=model_id.rstrip(b"\x00").decode(),
            tokenizer_id=tokenizer_id.rstrip(b"\x00").decode(),
            chunk_size=chunk_size,
            num_chunks=num_chunks,
            codec_per_chunk=codec_per_chunk,
            bitstream_offset=bitstream_offset,
        )
