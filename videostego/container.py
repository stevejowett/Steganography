"""On-the-wire blob that is embedded into the video (version 2, streaming).

    0   4B   magic "VSTG"
    4   1B   format version (2)
    5   1B   reserved (0)
    6   16B  scrypt salt
    22  8B   AES-GCM nonce base (per-chunk nonce = base || u32 chunk index)
    30  4B   plaintext chunk size
    34  1B   mode: high nibble = method (0 = DCT, 1 = LSB), low nibble = bits per unit
             (bits per 8x8 block for DCT, bits per byte for LSB)
    35  1B   repetition factor (1, 3 or 5)
    36  8B   body length (sum of all encrypted chunks incl. 16-byte tags)
    44  4B   CRC-32 of bytes 0..43
    48  var  body: chunk_0 || chunk_1 || ... (each = ciphertext || tag)
    end 4B   CRC-32 of the body (fast corruption check before decrypting)

Bytes 0..33 plus the chunk index and an is-last flag form the associated data
of every GCM chunk.  The embedding mode (bits/block, repetition) and the body
length are chosen *after* encryption, so they sit outside the AAD; they are
structural only and protected by the CRC.
"""

from __future__ import annotations

import os
import struct
import zlib
from dataclasses import dataclass
from typing import BinaryIO, Callable, Iterator, Optional

from . import crypto

MAGIC = b"VSTG"
VERSION = 2
HEADER_FMT = ">4sBB16s8sIBBQI"
HEADER_LEN = struct.calcsize(HEADER_FMT)  # 48
AAD_LEN = 34  # magic .. chunk size
TRAILER_LEN = 4


class NotStegoError(Exception):
    """Raised when no valid header can be found (not a stego video, or wrong password)."""


METHOD_DCT = 0
METHOD_LSB = 1


@dataclass
class Header:
    method: int
    bits: int
    repeat: int
    salt: bytes
    nonce_base: bytes
    chunk_size: int
    body_len: int
    aad: bytes

    @property
    def total_len(self) -> int:
        return HEADER_LEN + self.body_len + TRAILER_LEN

    @property
    def chunk_count(self) -> int:
        full = self.chunk_size + crypto.TAG_LEN
        return -(-self.body_len // full) if self.body_len else 0


def blob_length(record_len: int, chunk_size: int = crypto.DEFAULT_CHUNK_SIZE) -> int:
    """Size of the embedded blob for a record of ``record_len`` bytes."""
    chunks = max(1, -(-record_len // chunk_size))
    return HEADER_LEN + record_len + chunks * crypto.TAG_LEN + TRAILER_LEN


def _pack_header(salt: bytes, nonce_base: bytes, chunk_size: int,
                 method: int, bits: int, repeat: int, body_len: int) -> bytes:
    mode = ((method & 0x0F) << 4) | (bits & 0x0F)
    head = struct.pack(">4sBB16s8sIBBQ", MAGIC, VERSION, 0, salt, nonce_base, chunk_size,
                       mode, repeat, body_len)
    return head + struct.pack(">I", zlib.crc32(head) & 0xFFFFFFFF)


@dataclass
class PendingBlob:
    """An encrypted blob whose header still lacks the embedding mode."""
    salt: bytes
    nonce_base: bytes
    chunk_size: int
    body_len: int

    @property
    def total_len(self) -> int:
        return HEADER_LEN + self.body_len + TRAILER_LEN


def write_blob(record_pieces: Iterator[bytes], password: str, out: BinaryIO,
               chunk_size: int = crypto.DEFAULT_CHUNK_SIZE,
               progress: Optional[Callable[[int], None]] = None) -> PendingBlob:
    """Encrypt a record stream into ``out`` (seekable), chunk by chunk.

    Call :func:`finalize_blob` afterwards to write the header with the chosen
    embedding mode.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    salt = os.urandom(crypto.SALT_LEN)
    nonce_base = os.urandom(crypto.NONCE_BASE_LEN)
    placeholder = _pack_header(salt, nonce_base, chunk_size, 0, 0, 0, 0)
    enc = crypto.ChunkEncryptor(password, salt, nonce_base, placeholder[:AAD_LEN])

    out.seek(0)
    out.write(placeholder)
    body_len = 0
    crc = 0
    for chunk, is_last in crypto.iter_chunks(record_pieces, chunk_size):
        ct = enc.encrypt(chunk, is_last)
        out.write(ct)
        crc = zlib.crc32(ct, crc)
        body_len += len(ct)
        if progress:
            progress(body_len)
    out.write(struct.pack(">I", crc & 0xFFFFFFFF))
    out.flush()
    return PendingBlob(salt=salt, nonce_base=nonce_base, chunk_size=chunk_size, body_len=body_len)


def finalize_blob(out: BinaryIO, pending: PendingBlob, method: int, bits: int, repeat: int) -> int:
    """Write the final header; returns the total blob length."""
    if not 0 <= method <= 15 or not 1 <= bits <= 15 or not 1 <= repeat <= 255:
        raise ValueError("method / bits / repeat out of range")
    out.seek(0)
    out.write(_pack_header(pending.salt, pending.nonce_base, pending.chunk_size,
                           method, bits, repeat, pending.body_len))
    out.flush()
    return pending.total_len


def parse_header(buf: bytes) -> Header:
    if len(buf) < HEADER_LEN:
        raise ValueError("need more bytes")
    magic, version, _reserved, salt, nonce_base, chunk_size, mode, repeat, body_len, crc = \
        struct.unpack(HEADER_FMT, buf[:HEADER_LEN])
    if magic != MAGIC:
        raise NotStegoError("no hidden payload found (or wrong password)")
    if zlib.crc32(buf[: HEADER_LEN - 4]) & 0xFFFFFFFF != crc:
        raise NotStegoError("header checksum mismatch (corrupted video or wrong password)")
    if version != VERSION:
        raise NotStegoError(f"unsupported payload version {version}")
    if chunk_size == 0:
        raise NotStegoError("invalid chunk size in header")
    return Header(method=mode >> 4, bits=mode & 0x0F, repeat=repeat, salt=salt,
                  nonce_base=nonce_base, chunk_size=chunk_size, body_len=body_len,
                  aad=bytes(buf[:AAD_LEN]))


def read_blob(blob: BinaryIO, password: str, open_sink: Callable[[str, int], BinaryIO],
              progress: Optional[Callable[[int, int], None]] = None) -> crypto.RecordWriter:
    """Verify, decrypt and decompress a blob file, streaming the secret to ``open_sink``."""
    blob.seek(0)
    hdr = parse_header(blob.read(HEADER_LEN))

    # Pass 1: CRC-32 of the body (cheap, catches corruption before any crypto).
    crc = 0
    remaining = hdr.body_len
    while remaining:
        piece = blob.read(min(crypto.IO_BLOCK, remaining))
        if not piece:
            raise crypto.IntegrityError("blob is truncated")
        crc = zlib.crc32(piece, crc)
        remaining -= len(piece)
    trailer = blob.read(TRAILER_LEN)
    if len(trailer) < TRAILER_LEN:
        raise crypto.IntegrityError("blob trailer is missing")
    if (struct.unpack(">I", trailer)[0]) != (crc & 0xFFFFFFFF):
        raise crypto.IntegrityError("payload CRC-32 mismatch: video data is corrupted")

    # Pass 2: decrypt chunk by chunk.
    dec = crypto.ChunkDecryptor(password, hdr.salt, hdr.nonce_base, hdr.aad)
    writer = crypto.RecordWriter(open_sink)
    blob.seek(HEADER_LEN)
    full = hdr.chunk_size + crypto.TAG_LEN
    n = hdr.chunk_count
    if n == 0:
        raise crypto.IntegrityError("payload has no chunks")
    done = 0
    for i in range(n):
        size = full if i < n - 1 else hdr.body_len - full * (n - 1)
        ct = blob.read(size)
        if len(ct) != size:
            raise crypto.IntegrityError("blob is truncated")
        writer.feed(dec.decrypt(ct, is_last=(i == n - 1)))
        done += size
        if progress:
            progress(done, hdr.body_len)
    writer.finish()
    return writer
