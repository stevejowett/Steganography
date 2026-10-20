"""DCT-domain differential embedding in 8x8 luminance blocks.

Each 8x8 block of the Y plane carries ``bits_per_block`` bits (1..8). Bit ``p``
of a block is encoded in the *sign of the difference* between the two
mid-frequency DCT coefficients of pair ``PAIRS[p]``:

    bit 1  ->  C[a] - C[b] >=  +strength
    bit 0  ->  C[a] - C[b] <=  -strength

Differential embedding is self-referencing: the decoder needs no original
video, only the coefficients of the same block.  After embedding, the block is
inverse-transformed and rounded to 8-bit; a verify-and-boost loop re-reads the
rounded block and pushes the coefficients further apart for any block whose
margin was eroded by rounding/clipping.
"""

from __future__ import annotations

import numpy as np

BLOCK = 8
# Mid-frequency coefficient pairs ((row, col), (row, col)), symmetric about the
# diagonal so the natural difference is zero-mean. Pair 0 is the default; the
# later pairs are only used in the higher-capacity modes.
PAIRS = (
    ((3, 2), (2, 3)),
    ((4, 1), (1, 4)),
    ((3, 1), (1, 3)),
    ((4, 2), (2, 4)),
    ((2, 1), (1, 2)),
    ((5, 1), (1, 5)),
    ((4, 3), (3, 4)),
    ((5, 2), (2, 5)),
)
MAX_BITS_PER_BLOCK = len(PAIRS)

DEFAULT_STRENGTH = 10.0
MAX_BOOST_ITER = 8


def _dct_matrix(n: int = BLOCK) -> np.ndarray:
    k = np.arange(n)[:, None]
    i = np.arange(n)[None, :]
    m = np.sqrt(2.0 / n) * np.cos(np.pi * (2 * i + 1) * k / (2.0 * n))
    m[0, :] = np.sqrt(1.0 / n)
    return m.astype(np.float64)


_D = _dct_matrix()
_DT = _D.T.copy()


def block_count(height: int, width: int) -> int:
    return (height // BLOCK) * (width // BLOCK)


def _to_blocks(plane: np.ndarray) -> tuple[np.ndarray, int, int]:
    h, w = plane.shape
    bh, bw = h // BLOCK, w // BLOCK
    view = plane[: bh * BLOCK, : bw * BLOCK].reshape(bh, BLOCK, bw, BLOCK)
    blocks = view.transpose(0, 2, 1, 3).reshape(bh * bw, BLOCK, BLOCK)
    return blocks, bh, bw


def _from_blocks(blocks: np.ndarray, bh: int, bw: int) -> np.ndarray:
    return blocks.reshape(bh, bw, BLOCK, BLOCK).transpose(0, 2, 1, 3).reshape(bh * BLOCK, bw * BLOCK)


def _dct2(blocks: np.ndarray) -> np.ndarray:
    return _D @ blocks @ _DT


def _idct2(coefs: np.ndarray) -> np.ndarray:
    return _DT @ coefs @ _D


def _check_k(bits_per_block: int) -> None:
    if not 1 <= bits_per_block <= MAX_BITS_PER_BLOCK:
        raise ValueError(f"bits_per_block must be 1..{MAX_BITS_PER_BLOCK}")


def embed_bits(plane: np.ndarray, bits: np.ndarray, positions: np.ndarray,
               strength: float = DEFAULT_STRENGTH, bits_per_block: int = 1) -> np.ndarray:
    """Embed ``bits`` into the blocks listed in ``positions`` of an 8-bit Y plane.

    ``positions`` is the full keyed permutation of all blocks (length n).  The
    bit stream is laid out pair-major: bits[0:n] go to pair 0 of blocks
    positions[0..n), bits[n:2n] to pair 1 of the same blocks, and so on, so
    ``len(bits) <= n * bits_per_block``.  Returns a new uint8 plane.
    """
    _check_k(bits_per_block)
    n = len(positions)
    k = bits_per_block
    if len(bits) > n * k:
        raise ValueError("too many bits for this plane")
    if len(bits) == 0:
        return plane.copy()

    out = plane.copy()
    blocks, bh, bw = _to_blocks(out)
    m = min(n, len(bits))  # pair 0 always carries the most bits
    sel = blocks[positions[:m]].astype(np.float64)

    signs = np.zeros((k, m), dtype=np.float64)
    mask = np.zeros((k, m), dtype=bool)
    for p in range(k):
        cnt = int(np.clip(len(bits) - p * n, 0, m))
        if cnt == 0:
            break
        chunk = bits[p * n: p * n + cnt]
        signs[p, :cnt] = np.where(chunk.astype(bool), 1.0, -1.0)
        mask[p, :cnt] = True

    coefs = _dct2(sel)
    for p in range(k):
        (a0, a1), (b0, b1) = PAIRS[p]
        diff = coefs[:, a0, a1] - coefs[:, b0, b1]
        delta = np.where(mask[p], np.maximum(0.0, strength - signs[p] * diff) / 2.0, 0.0)
        coefs[:, a0, a1] += signs[p] * delta
        coefs[:, b0, b1] -= signs[p] * delta
    pix = np.clip(np.rint(_idct2(coefs)), 0, 255)

    # Verify-and-boost: rounding / clipping may have eroded the margin.
    for _ in range(MAX_BOOST_ITER):
        check = _dct2(pix)
        bad_any = np.zeros(m, dtype=bool)
        extras = np.zeros((k, m), dtype=np.float64)
        for p in range(k):
            (a0, a1), (b0, b1) = PAIRS[p]
            margin = signs[p] * (check[:, a0, a1] - check[:, b0, b1])
            bad = mask[p] & (margin < strength * 0.5)
            extras[p] = np.where(bad, (strength - margin) / 2.0 + 1.0, 0.0)
            bad_any |= bad
        if not bad_any.any():
            break
        idx = np.nonzero(bad_any)[0]
        c = check[idx]
        for p in range(k):
            (a0, a1), (b0, b1) = PAIRS[p]
            c[:, a0, a1] += signs[p, idx] * extras[p, idx]
            c[:, b0, b1] -= signs[p, idx] * extras[p, idx]
        pix[idx] = np.clip(np.rint(_idct2(c)), 0, 255)

    blocks[positions[:m]] = pix.astype(np.uint8)
    out[: bh * BLOCK, : bw * BLOCK] = _from_blocks(blocks, bh, bw)
    return out


def extract_bits(plane: np.ndarray, positions: np.ndarray,
                 bits_per_block: int = MAX_BITS_PER_BLOCK) -> np.ndarray:
    """Read bits from every block in ``positions``.

    Returns a uint8 array of shape (bits_per_block, len(positions)); row ``p``
    holds the bit stored in coefficient pair ``p`` of each block.
    """
    _check_k(bits_per_block)
    blocks, _, _ = _to_blocks(plane)
    coefs = _dct2(blocks[positions].astype(np.float64))
    rows = []
    for p in range(bits_per_block):
        (a0, a1), (b0, b1) = PAIRS[p]
        rows.append((coefs[:, a0, a1] - coefs[:, b0, b1]) > 0)
    return np.stack(rows).astype(np.uint8)
