"""Bit-level helpers: repetition coding, majority vote, keyed block permutation."""

from __future__ import annotations

import hashlib
import struct

import numpy as np

REPEAT = 5               # default: each payload bit is written 5 times
REPEAT_OPTIONS = (5, 3, 1)  # allowed repetition factors (decoder tries all of them)


def bytes_to_bits(data: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8))


def bits_to_bytes(bits: np.ndarray) -> bytes:
    n = (len(bits) // 8) * 8
    return np.packbits(bits[:n]).tobytes()


def repeat_bits(bits: np.ndarray, r: int = REPEAT) -> np.ndarray:
    if r == 1:
        return bits.astype(np.uint8)
    return np.repeat(bits.astype(np.uint8), r)


def majority_vote(bits: np.ndarray, r: int = REPEAT) -> np.ndarray:
    if r == 1:
        return bits.astype(np.uint8)
    n = (len(bits) // r) * r
    groups = bits[:n].reshape(-1, r)
    return (groups.sum(axis=1) * 2 > r).astype(np.uint8)


def frame_permutation(seed: bytes, frame_index: int, n_blocks: int) -> np.ndarray:
    """Keyed, per-frame permutation of block indices.

    The same (password, frame index) always yields the same ordering, so the
    decoder can replay it; without the password the bit positions look random.
    """
    digest = hashlib.sha256(seed + struct.pack(">Q", frame_index)).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    return rng.permutation(n_blocks)
