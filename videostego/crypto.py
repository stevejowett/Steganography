"""Password-based AES-256-GCM encryption (chunked / streaming) with zlib compression.

The *record* is the plaintext stream that gets encrypted:

    u16  filename length
    var  filename (UTF-8)
    u64  original file size
    var  zlib-compressed file contents
    32B  SHA-256 of the original file   (written last so one pass suffices)

The record is cut into ``chunk_size`` pieces and every piece is sealed with
AES-256-GCM under its own nonce (8-byte random base || 4-byte chunk index).
The chunk index and an ``is_last`` flag are part of the associated data, so
chunks cannot be reordered, dropped or truncated without detection.
"""

from __future__ import annotations

import hashlib
import os
import struct
import zlib
from typing import BinaryIO, Callable, Iterator, Optional

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

SALT_LEN = 16
NONCE_BASE_LEN = 8
TAG_LEN = 16
KEY_LEN = 32  # AES-256
DEFAULT_CHUNK_SIZE = 1 << 20  # 1 MiB of plaintext per GCM chunk
IO_BLOCK = 1 << 16

# scrypt parameters: ~32 MiB of memory, interactive-speed derivation.
SCRYPT_N = 2 ** 15
SCRYPT_R = 8
SCRYPT_P = 1


class WrongPasswordError(Exception):
    """Raised when a GCM tag does not verify (wrong password or corrupted data)."""


class IntegrityError(Exception):
    """Raised when a hash / checksum does not match."""


def derive_key(password: str, salt: bytes) -> bytes:
    """Derive a 256-bit AES key from a password using scrypt."""
    kdf = Scrypt(salt=salt, length=KEY_LEN, n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P)
    return kdf.derive(password.encode("utf-8"))


def permutation_seed(password: str) -> bytes:
    """Secret used to seed the per-frame block permutation.

    Derived from the password only (no salt), because the decoder must be
    able to locate the header *before* it can read the salt stored in it.
    """
    return hashlib.sha256(b"videostego/permutation/v1\x00" + password.encode("utf-8")).digest()


def _chunk_aad(base_aad: bytes, index: int, is_last: bool) -> bytes:
    return base_aad + struct.pack(">IB", index, 1 if is_last else 0)


def _chunk_nonce(nonce_base: bytes, index: int) -> bytes:
    return nonce_base + struct.pack(">I", index)


class ChunkEncryptor:
    def __init__(self, password: str, salt: bytes, nonce_base: bytes, base_aad: bytes):
        self._aead = AESGCM(derive_key(password, salt))
        self._nonce_base = nonce_base
        self._aad = base_aad
        self.index = 0

    def encrypt(self, plaintext: bytes, is_last: bool) -> bytes:
        ct = self._aead.encrypt(_chunk_nonce(self._nonce_base, self.index), plaintext,
                                _chunk_aad(self._aad, self.index, is_last))
        self.index += 1
        return ct


class ChunkDecryptor:
    def __init__(self, password: str, salt: bytes, nonce_base: bytes, base_aad: bytes):
        self._aead = AESGCM(derive_key(password, salt))
        self._nonce_base = nonce_base
        self._aad = base_aad
        self.index = 0

    def decrypt(self, ciphertext: bytes, is_last: bool) -> bytes:
        try:
            pt = self._aead.decrypt(_chunk_nonce(self._nonce_base, self.index), ciphertext,
                                    _chunk_aad(self._aad, self.index, is_last))
        except InvalidTag as exc:
            if self.index == 0:
                raise WrongPasswordError("authentication failed: wrong password or corrupted payload") from exc
            raise IntegrityError(f"authentication failed on chunk {self.index}: payload is corrupted") from exc
        self.index += 1
        return pt


# --------------------------------------------------------------------------------------
# Record (plaintext) stream
# --------------------------------------------------------------------------------------

def record_prefix(name: str, size: int) -> bytes:
    name_b = os.path.basename(name).encode("utf-8")
    if len(name_b) > 0xFFFF:
        raise ValueError("filename too long")
    return struct.pack(">H", len(name_b)) + name_b + struct.pack(">Q", size)


def iter_record(path: str, level: int = 6) -> Iterator[bytes]:
    """Yield the record for ``path`` piece by piece without loading the file."""
    size = os.path.getsize(path)
    yield record_prefix(path, size)
    comp = zlib.compressobj(level)
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(IO_BLOCK)
            if not block:
                break
            sha.update(block)
            out = comp.compress(block)
            if out:
                yield out
    yield comp.flush()
    yield sha.digest()


def record_length(path: str, level: int = 6) -> int:
    """Length of the record for ``path`` (compresses the file once, streaming)."""
    size = os.path.getsize(path)
    total = len(record_prefix(path, size)) + 32
    comp = zlib.compressobj(level)
    with open(path, "rb") as f:
        while True:
            block = f.read(IO_BLOCK)
            if not block:
                break
            total += len(comp.compress(block))
    total += len(comp.flush())
    return total


def iter_chunks(pieces: Iterator[bytes], chunk_size: int) -> Iterator[tuple[bytes, bool]]:
    """Re-cut a byte stream into ``chunk_size`` pieces, flagging the final one."""
    buf = bytearray()
    pending: Optional[bytes] = None
    for piece in pieces:
        buf.extend(piece)
        while len(buf) >= chunk_size:
            if pending is not None:
                yield pending, False
            pending = bytes(buf[:chunk_size])
            del buf[:chunk_size]
    if pending is not None and len(buf) == 0:
        yield pending, True
        return
    if pending is not None:
        yield pending, False
    yield bytes(buf), True


class RecordWriter:
    """Consume decrypted record bytes, stream the file to disk, verify SHA-256.

    ``open_sink(name, size)`` is called once the filename is known and must
    return a writable binary file object.
    """

    def __init__(self, open_sink: Callable[[str, int], BinaryIO]):
        self._open_sink = open_sink
        self._head = bytearray()
        self._sink: Optional[BinaryIO] = None
        self._decomp = zlib.decompressobj()
        self._sha = hashlib.sha256()
        self._tail = bytearray()
        self.name = ""
        self.size = 0
        self.written = 0
        self.digest = b""

    def feed(self, data: bytes) -> None:
        if self._sink is None:
            self._head.extend(data)
            if len(self._head) < 2:
                return
            (name_len,) = struct.unpack(">H", self._head[:2])
            if len(self._head) < 2 + name_len + 8:
                return
            self.name = bytes(self._head[2: 2 + name_len]).decode("utf-8", errors="replace")
            (self.size,) = struct.unpack(">Q", self._head[2 + name_len: 2 + name_len + 8])
            self._sink = self._open_sink(self.name, self.size)
            data = bytes(self._head[2 + name_len + 8:])
            self._head = bytearray()
            if not data:
                return
        if self._decomp.eof:
            self._tail.extend(data)
            return
        try:
            out = self._decomp.decompress(data)
        except zlib.error as exc:
            raise IntegrityError(f"decompression failed: {exc}") from exc
        if out:
            self._sink.write(out)
            self._sha.update(out)
            self.written += len(out)
        if self._decomp.eof:
            self._tail.extend(self._decomp.unused_data)

    def finish(self) -> bytes:
        """Verify size and SHA-256; returns the digest."""
        if self._sink is None:
            raise IntegrityError("record truncated before the filename")
        if not self._decomp.eof:
            raise IntegrityError("compressed stream is truncated")
        self._sink.flush()
        if self.written != self.size:
            raise IntegrityError(f"size mismatch: expected {self.size} bytes, got {self.written}")
        if len(self._tail) != 32:
            raise IntegrityError("record trailer (SHA-256) is malformed")
        digest = self._sha.digest()
        if digest != bytes(self._tail):
            raise IntegrityError("SHA-256 mismatch: extracted file is corrupted")
        self.digest = digest
        return digest
