"""High-level hide / extract pipeline (streaming; memory use is independent of file size).

hide():    secret file -> zlib + chunked AES-256-GCM into a temp blob file
           -> pick embedding mode -> bits x repeat -> keyed positions
           -> embed into the leading frames (LSB bit-planes or DCT coefficients)
           -> data frames lossless H.264 (QP 0), remaining frames lossy
           -> remux with the carrier's audio.

extract(): video -> frames -> detect mode from the header -> majority vote
           -> temp blob file -> CRC / GCM / SHA-256 verification -> original file.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from typing import BinaryIO, Callable, Optional

import numpy as np

from . import bitstream, container, crypto, dct, ffmpeg_io, lsb

ProgressFn = Callable[[int, int, str], None]  # (done, total, message)
AUTO = 0  # "choose automatically" for bits / repeat

METHOD_DCT = container.METHOD_DCT
METHOD_LSB = container.METHOD_LSB
METHOD_NAMES = {METHOD_DCT: "dct", METHOD_LSB: "lsb"}
METHODS_BY_NAME = {v: k for k, v in METHOD_NAMES.items()}
AUTO_MAX_FRACTION = 0.10  # Auto tries to keep lossless (data) frames below this share


class CapacityError(Exception):
    pass


def max_bits(method: int) -> int:
    return lsb.MAX_BITS_PER_BYTE if method == METHOD_LSB else dct.MAX_BITS_PER_BLOCK


def slots_per_frame(method: int, width: int, height: int) -> int:
    """Embedding units per frame: bytes for LSB, 8x8 luma blocks for DCT."""
    if method == METHOD_LSB:
        return ffmpeg_io.frame_size(width, height)
    return dct.block_count(height, width)


@dataclass(frozen=True)
class Capacity:
    method: int
    bits: int
    repeat: int
    slots: int
    frames: int

    @property
    def raw_bits_per_frame(self) -> int:
        return self.slots * self.bits

    @property
    def bytes_total(self) -> int:
        return (self.raw_bits_per_frame * self.frames) // (8 * self.repeat)

    @property
    def bytes_per_frame(self) -> float:
        return self.raw_bits_per_frame / (8.0 * self.repeat)

    @property
    def label(self) -> str:
        unit = "bit(s)/byte" if self.method == METHOD_LSB else "bit(s)/block"
        return f"{METHOD_NAMES[self.method].upper()} {self.bits} {unit}, {self.repeat}x repetition"


def capacity_for(info: ffmpeg_io.VideoInfo, method: int = METHOD_LSB, bits: int = 1,
                 repeat: int = 1) -> Capacity:
    if bits == AUTO:
        bits = max_bits(method)
    if repeat == AUTO:
        repeat = 1
    return Capacity(method=method, bits=bits, repeat=repeat,
                    slots=slots_per_frame(method, info.even_width, info.even_height),
                    frames=info.frame_estimate)


def frames_needed(cap: Capacity, payload_bytes: int) -> int:
    """Number of data-carrying (lossless) frames for ``payload_bytes`` in this mode."""
    return -(-payload_bytes * 8 * cap.repeat // max(cap.raw_bits_per_frame, 1))


def choose_mode(info: ffmpeg_io.VideoInfo, payload_bytes: int, method: int = METHOD_LSB,
                bits: int = AUTO, repeat: int = AUTO,
                max_fraction: float = AUTO_MAX_FRACTION) -> Optional[Capacity]:
    """Resolve AUTO settings for ``method``; None if the payload does not fit at all.

    Data frames are stored losslessly and dominate the output size, so Auto
    prefers the least visible settings that keep them under ``max_fraction``
    of the video, and otherwise falls back to the fewest-frames setting.
    LSB mode needs no repetition (lossless frames are exact), so Auto uses 1x.
    """
    bits_options = [bits] if bits != AUTO else list(range(1, max_bits(method) + 1))
    if repeat != AUTO:
        repeat_options = [repeat]
    else:
        repeat_options = [1] if method == METHOD_LSB else list(bitstream.REPEAT_OPTIONS)
    fitting = []
    for b in bits_options:
        for r in repeat_options:
            cap = capacity_for(info, method, b, r)
            if cap.bytes_total >= payload_bytes:
                n = frames_needed(cap, payload_bytes)
                fitting.append((n, b, r, cap))
                if n <= max_fraction * cap.frames:
                    return cap
    if not fitting:
        return None
    return min(fitting, key=lambda t: (t[0], t[1], -t[2]))[3]


def estimate_output_bytes(info: ffmpeg_io.VideoInfo, cap: Capacity, payload_bytes: int,
                          carrier_bytes: int, lossless_all: bool = False) -> int:
    """Rough output size: lossless frames at ~raw/4 plus the embedded bits, rest at source bitrate."""
    raw_frame = ffmpeg_io.frame_size(info.even_width, info.even_height)
    frames = cap.frames if lossless_all else min(frames_needed(cap, payload_bytes), cap.frames)
    lossless = frames * (raw_frame / 4.0) + payload_bytes * cap.repeat
    rest = 0.0 if lossless_all else carrier_bytes * (1 - frames / max(cap.frames, 1))
    return int(lossless + rest)


def payload_size(secret_path: str, chunk_size: int = crypto.DEFAULT_CHUNK_SIZE) -> int:
    """Size in bytes of the blob that would be embedded for ``secret_path`` (streaming)."""
    return container.blob_length(crypto.record_length(secret_path), chunk_size)


def _noop(_done: int, _total: int, _msg: str) -> None:
    pass


def _temp_file(directory: str, suffix: str) -> str:
    os.makedirs(directory, exist_ok=True)
    fd, path = tempfile.mkstemp(prefix=".stego_", suffix=suffix, dir=directory)
    os.close(fd)
    return path


def _remove(path: Optional[str]) -> None:
    if path:
        try:
            os.remove(path)
        except OSError:
            pass


class _BlobBitSource:
    """Stream the raw (repeated) bit sequence of a blob file, piece by piece."""

    def __init__(self, f: BinaryIO, repeat: int):
        self._f = f
        self._r = repeat
        self._buf = np.empty(0, dtype=np.uint8)
        self._eof = False

    @property
    def exhausted(self) -> bool:
        return self._eof and len(self._buf) == 0

    def take(self, n: int) -> np.ndarray:
        # Read ahead until *more* than n bits are buffered (or EOF is seen) so
        # that ``exhausted`` is accurate immediately after the last take().
        while len(self._buf) <= n and not self._eof:
            data = self._f.read(crypto.IO_BLOCK * 4)
            if not data:
                self._eof = True
                break
            raw = bitstream.repeat_bits(bitstream.bytes_to_bits(data), self._r)
            self._buf = np.concatenate((self._buf, raw)) if len(self._buf) else raw
        out = self._buf[:n]
        self._buf = self._buf[n:]
        return out


@dataclass
class HideResult:
    output: str
    payload_bytes: int
    method: int
    bits: int
    repeat: int
    frames_used: int
    frames_total: int
    lossless_all: bool
    audio_note: str

    @property
    def mode_label(self) -> str:
        unit = "bit(s)/byte" if self.method == METHOD_LSB else "bit(s)/block"
        return f"{METHOD_NAMES[self.method].upper()} {self.bits} {unit}, {self.repeat}x repetition"


def _embed_frame(method: int, frame: bytearray, chunk: np.ndarray, perm: np.ndarray,
                 bits: int, strength: float, w: int, h: int, verify: bool) -> int:
    """Embed ``chunk`` into ``frame`` in place; returns the number of unreadable bits."""
    if method == METHOD_LSB:
        lsb.embed_bits(frame, chunk, perm, bits)
        if not verify:
            return 0
        got = lsb.extract_bits(frame, perm, bits).ravel()[: len(chunk)]
        return int((got != chunk).sum())
    y_len = w * h
    y = np.frombuffer(frame, dtype=np.uint8, count=y_len).reshape(h, w)
    y_new = dct.embed_bits(y, chunk, perm, strength, bits_per_block=bits)
    frame[:y_len] = y_new.tobytes()
    if not verify:
        return 0
    got = dct.extract_bits(y_new, perm, bits).ravel()[: len(chunk)]
    return int((got != chunk).sum())


def hide(carrier: str, secret: str, output: str, password: str,
         method: int = METHOD_LSB, bits: int = AUTO, repeat: int = AUTO,
         strength: float = dct.DEFAULT_STRENGTH, lossless_all: bool = False,
         crf: Optional[int] = None, preset: str = "fast",
         chunk_size: int = crypto.DEFAULT_CHUNK_SIZE, progress: ProgressFn = _noop) -> HideResult:
    if not password:
        raise ValueError("password must not be empty")
    if os.path.abspath(output) == os.path.abspath(carrier):
        raise ValueError("output path must differ from the carrier video")
    if method not in METHOD_NAMES:
        raise ValueError("unknown method")
    if bits != AUTO and not 1 <= bits <= max_bits(method):
        raise ValueError(f"bits must be 0 (auto) or 1..{max_bits(method)} for {METHOD_NAMES[method]}")
    if repeat != AUTO and repeat not in bitstream.REPEAT_OPTIONS:
        raise ValueError(f"repeat must be 0 (auto) or one of {bitstream.REPEAT_OPTIONS}")
    if crf is not None and not 0 <= crf <= 51:
        raise ValueError("crf must be 0..51")
    if not os.path.isfile(secret):
        raise FileNotFoundError(secret)
    info = ffmpeg_io.probe(carrier)
    w, h = info.even_width, info.even_height
    if w < dct.BLOCK or h < dct.BLOCK:
        raise ValueError("carrier video is too small")

    out_dir = os.path.dirname(os.path.abspath(output)) or "."
    blob_path = _temp_file(out_dir, ".blob")
    tmp_video: Optional[str] = None
    try:
        # 1. Compress + encrypt the secret into a temp blob (streaming, one pass).
        secret_size = os.path.getsize(secret)
        progress(0, 1, "Compressing and encrypting payload...")
        with open(blob_path, "w+b") as bf:
            pending = container.write_blob(
                crypto.iter_record(secret), password, bf,
                chunk_size=chunk_size,
                progress=lambda done: progress(min(done, secret_size), max(secret_size, 1),
                                               f"Encrypting payload... {done / 2**20:.1f} MB"))
            blob_len = pending.total_len

            # 2. Pick the embedding mode and check the capacity.
            cap = choose_mode(info, blob_len, method, bits, repeat)
            if cap is None:
                shown = capacity_for(info, method, bits, repeat)
                best = capacity_for(info, method, max_bits(method), 1)
                raise CapacityError(
                    f"payload is {blob_len:,} bytes but the carrier holds only about "
                    f"{shown.bytes_total:,} bytes at {shown.label} "
                    f"({shown.bytes_per_frame:,.0f} bytes/frame x ~{shown.frames} frames). "
                    f"Maximum for this carrier with {METHOD_NAMES[method].upper()} is about "
                    f"{best.bytes_total:,} bytes. Use a longer or larger video, or a smaller file.")
            container.finalize_blob(bf, pending, cap.method, cap.bits, cap.repeat)

            # 3. Embed frame by frame. Only the leading data-carrying frames are
            #    encoded losslessly; the rest are encoded at the source bitrate / CRF.
            seed = crypto.permutation_seed(password)
            per_frame = cap.raw_bits_per_frame
            total_raw = blob_len * 8 * cap.repeat
            need_frames = -(-total_raw // per_frame)
            tmp_video = _temp_file(out_dir, ".h264")
            writer = ffmpeg_io.SegmentedWriter(
                tmp_video, w, h, info.fps,
                lossless_frames=None if lossless_all else need_frames,
                crf=crf, bitrate_kbps=info.bitrate_kbps, preset=preset)
            frames_written = 0
            frames_used = 0
            pos = 0
            bf.seek(0)
            source = _BlobBitSource(bf, cap.repeat)
            try:
                for frame in ffmpeg_io.read_frames(carrier, w, h):
                    if not source.exhausted:
                        chunk = source.take(per_frame)
                        if len(chunk):
                            perm = bitstream.frame_permutation(seed, frames_written, cap.slots)
                            bad = _embed_frame(cap.method, frame, chunk, perm, cap.bits, strength,
                                               w, h, verify=(cap.repeat == 1))
                            if bad:
                                raise RuntimeError(
                                    f"{bad} bit(s) could not be embedded reliably in frame "
                                    f"{frames_written}; raise the strength or use repetition")
                            pos += len(chunk)
                            frames_used += 1
                    writer.write(frame)
                    frames_written += 1
                    if frames_written % 10 == 0:
                        total = max(cap.frames, frames_written)
                        progress(frames_written, total,
                                 f"Encoding frame {frames_written}/{total}"
                                 + (f"  (embedding, {100 * pos // total_raw}%)" if not source.exhausted
                                    else "  (lossless)" if lossless_all else "  (pass-through)"))
                writer.close()
            except BaseException:
                writer.abort()
                tmp_video = None
                raise

        if not source.exhausted:
            raise CapacityError(
                f"carrier ended after {frames_written} frames; only {100 * pos // total_raw}% of "
                f"the payload fitted. Use a longer video or a smaller file.")

        # 4. Remux with the carrier's audio into the requested container.
        progress(frames_written, frames_written, "Muxing audio...")
        note = ffmpeg_io.mux_with_audio(tmp_video, carrier, output, fps=info.fps)
        if not info.has_audio:
            note = "carrier has no audio track"
    finally:
        _remove(blob_path)
        _remove(tmp_video)

    progress(frames_written, frames_written, "Done")
    return HideResult(output=output, payload_bytes=blob_len, method=cap.method, bits=cap.bits,
                      repeat=cap.repeat, frames_used=frames_used, frames_total=frames_written,
                      lossless_all=lossless_all, audio_note=note)


@dataclass
class ExtractResult:
    output: str
    name: str
    size: int
    sha256_hex: str
    method: int
    bits: int
    repeat: int
    frames_read: int

    @property
    def mode_label(self) -> str:
        unit = "bit(s)/byte" if self.method == METHOD_LSB else "bit(s)/block"
        return f"{METHOD_NAMES[self.method].upper()} {self.bits} {unit}, {self.repeat}x repetition"


class _VotedSink:
    """Majority-vote a raw bit stream (spanning frames) and pack it into a file."""

    def __init__(self, f: BinaryIO, repeat: int, limit: int):
        self._f = f
        self._r = repeat
        self._limit = limit  # voted bits wanted in total
        self._raw_carry = np.empty(0, dtype=np.uint8)
        self._bit_carry = np.empty(0, dtype=np.uint8)
        self.count = 0  # voted bits so far

    @property
    def done(self) -> bool:
        return self.count >= self._limit

    def feed(self, raw: np.ndarray) -> None:
        if self.done:
            return
        buf = np.concatenate((self._raw_carry, raw)) if len(self._raw_carry) else raw
        m = (len(buf) // self._r) * self._r
        self._raw_carry = buf[m:].copy()
        voted = bitstream.majority_vote(buf[:m], self._r)
        voted = voted[: self._limit - self.count]
        self.count += len(voted)
        bits = np.concatenate((self._bit_carry, voted)) if len(self._bit_carry) else voted
        nb = (len(bits) // 8) * 8
        if nb:
            self._f.write(np.packbits(bits[:nb]).tobytes())
        self._bit_carry = bits[nb:].copy()

    def flush(self) -> None:
        if len(self._bit_carry):
            self._f.write(np.packbits(self._bit_carry).tobytes())
            self._bit_carry = np.empty(0, dtype=np.uint8)
        self._f.flush()


# Candidate (method, repeat options) tried on the first frames, in order.
_DETECT_ORDER = ((METHOD_LSB, (1, 3, 5)), (METHOD_DCT, bitstream.REPEAT_OPTIONS))


def _detect_mode(pending: dict[int, list[np.ndarray]]) -> Optional[container.Header]:
    """Try every (method, repeat, bits) combination on the first frames.

    ``pending[method]`` holds one (max_bits, slots) array per frame. Returns
    the header once a combination validates, None if more frames are needed,
    and raises NotStegoError once every combination had enough bits and none matched.
    """
    last_error: Exception = container.NotStegoError("no hidden payload found (or wrong password)")
    tried_all = True
    for method, repeats in _DETECT_ORDER:
        frames = pending[method]
        for r in repeats:
            header_bits = container.HEADER_LEN * 8 * r
            for b in range(1, max_bits(method) + 1):
                stream = np.concatenate([e[:b].ravel() for e in frames])
                if len(stream) < header_bits:
                    tried_all = False
                    continue
                hdr_bytes = bitstream.bits_to_bytes(bitstream.majority_vote(stream[:header_bits], r))
                try:
                    hdr = container.parse_header(hdr_bytes)
                except container.NotStegoError as exc:
                    last_error = exc
                    continue
                if hdr.method == method and hdr.bits == b and hdr.repeat == r:
                    return hdr
    if tried_all:
        raise last_error
    return None


def extract(stego_video: str, password: str, output_dir: str,
            progress: ProgressFn = _noop, overwrite: bool = False) -> ExtractResult:
    if not password:
        raise ValueError("password must not be empty")
    info = ffmpeg_io.probe(stego_video)
    w, h = info.even_width, info.even_height
    seed = crypto.permutation_seed(password)
    y_len = w * h
    slots = {m: slots_per_frame(m, w, h) for m in METHOD_NAMES}

    os.makedirs(output_dir, exist_ok=True)
    blob_path = _temp_file(output_dir, ".blob")
    out_path: Optional[str] = None
    out_file: Optional[BinaryIO] = None
    try:
        pending: dict[int, list[np.ndarray]] = {m: [] for m in METHOD_NAMES}
        header: Optional[container.Header] = None
        sink: Optional[_VotedSink] = None
        frames_read = 0

        def read_frame(method: int, frame: bytearray, perm: np.ndarray, nbits: int) -> np.ndarray:
            if method == METHOD_LSB:
                return lsb.extract_bits(frame, perm, nbits)
            y = np.frombuffer(frame, dtype=np.uint8, count=y_len).reshape(h, w)
            return dct.extract_bits(y, perm, nbits)

        with open(blob_path, "w+b") as bf:
            progress(0, 1, "Scanning video...")
            for frame in ffmpeg_io.read_frames(stego_video, w, h):
                if header is None:
                    for m in METHOD_NAMES:
                        perm = bitstream.frame_permutation(seed, frames_read, slots[m])
                        pending[m].append(read_frame(m, frame, perm, max_bits(m)))
                    frames_read += 1
                    header = _detect_mode(pending)  # raises NotStegoError when hopeless
                    if header is None:
                        continue
                    sink = _VotedSink(bf, header.repeat, header.total_len * 8)
                    for e in pending[header.method]:
                        sink.feed(e[:header.bits].ravel())
                    pending = {m: [] for m in METHOD_NAMES}
                else:
                    assert sink is not None
                    perm = bitstream.frame_permutation(seed, frames_read, slots[header.method])
                    frames_read += 1
                    sink.feed(read_frame(header.method, frame, perm, header.bits).ravel())

                est_frames = -(-header.total_len * 8 * header.repeat
                               // (slots[header.method] * header.bits))
                progress(min(frames_read, est_frames), est_frames,
                         f"Reading frame {frames_read}/{est_frames}")
                if sink.done:
                    break

            if header is None or sink is None:
                raise container.NotStegoError(
                    "video too short to contain a payload header (or wrong password)")
            if not sink.done:
                raise crypto.IntegrityError(
                    f"video ended after {frames_read} frames; payload is truncated")
            sink.flush()

            # Decrypt + decompress straight to the output file.
            def open_sink(name: str, _size: int) -> BinaryIO:
                nonlocal out_path, out_file
                safe = os.path.basename(name).replace("\\", "_").replace("/", "_") or "extracted.bin"
                path = os.path.join(output_dir, safe)
                if not overwrite:
                    base, ext = os.path.splitext(path)
                    i = 1
                    while os.path.exists(path):
                        path = f"{base} ({i}){ext}"
                        i += 1
                out_path = path
                out_file = open(path, "wb")
                return out_file

            progress(0, 1, "Verifying and decrypting...")
            writer = container.read_blob(
                bf, password, open_sink,
                progress=lambda d, t: progress(d, t, f"Decrypting... {100 * d // max(t, 1)}%"))
            out_file.close()
            out_file = None
    except BaseException:
        if out_file is not None:
            out_file.close()
        _remove(out_path)
        raise
    finally:
        _remove(blob_path)

    progress(frames_read, frames_read, "Done")
    return ExtractResult(output=out_path, name=writer.name, size=writer.size,
                         sha256_hex=writer.digest.hex(), method=header.method, bits=header.bits,
                         repeat=header.repeat, frames_read=frames_read)
