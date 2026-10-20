"""Compact spatial-domain embedding: bits in the low bit-planes of every frame byte.

Only frames that carry data are stored losslessly, so the low bits survive
exactly and no error tolerance is needed. The payload is spread over all
bytes of the yuv420p frame (Y, U and V planes) in a password-keyed order.
Each byte carries ``bits_per_byte`` bits (1..4): 1-2 bits are invisible,
3-4 add faint noise. Cost in the lossless stream is close to one coded bit
per payload bit, which is roughly 15x cheaper than the DCT method.
"""

from __future__ import annotations

import numpy as np

MAX_BITS_PER_BYTE = 4


def _check(bits_per_byte: int) -> None:
    if not 1 <= bits_per_byte <= MAX_BITS_PER_BYTE:
        raise ValueError(f"bits_per_byte must be 1..{MAX_BITS_PER_BYTE}")


def embed_bits(frame: bytearray, bits: np.ndarray, positions: np.ndarray,
               bits_per_byte: int = 1) -> None:
    """Write ``bits`` into ``frame`` in place.

    ``positions`` is the full keyed permutation of all byte offsets (length n).
    Layout is plane-major like the DCT method: bits[0:n] go to bit-plane 0 of
    bytes positions[0..n), bits[n:2n] to bit-plane 1, and so on.
    """
    _check(bits_per_byte)
    n = len(positions)
    if len(bits) > n * bits_per_byte:
        raise ValueError("too many bits for this frame")
    arr = np.frombuffer(frame, dtype=np.uint8)
    for p in range(bits_per_byte):
        cnt = int(np.clip(len(bits) - p * n, 0, n))
        if cnt == 0:
            break
        idx = positions[:cnt]
        vals = bits[p * n: p * n + cnt].astype(np.uint8)
        mask = np.uint8(0xFF ^ (1 << p))
        arr[idx] = (arr[idx] & mask) | (vals << np.uint8(p))


def extract_bits(frame: bytes | bytearray, positions: np.ndarray,
                 bits_per_byte: int = MAX_BITS_PER_BYTE, limit: int | None = None) -> np.ndarray:
    """Read the low bit-planes of the bytes listed in ``positions``.

    Returns a uint8 array of shape (bits_per_byte, len(positions)); row ``p``
    holds bit-plane ``p``. ``limit`` restricts the read to the first ``limit``
    positions (for a cheap header probe).
    """
    _check(bits_per_byte)
    arr = np.frombuffer(frame, dtype=np.uint8)
    pos = positions if limit is None else positions[:limit]
    sel = arr[pos]
    return np.stack([(sel >> np.uint8(p)) & np.uint8(1) for p in range(bits_per_byte)]).astype(np.uint8)
