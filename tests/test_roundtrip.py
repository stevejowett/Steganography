"""End-to-end tests: synthesize a carrier with audio, hide files, extract them.

Run with:  venv\\Scripts\\python tests\\test_roundtrip.py
(or:       venv\\Scripts\\python -m pytest tests)
"""

from __future__ import annotations

import io
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from videostego import bitstream, container, crypto, dct, ffmpeg_io, lsb, stego  # noqa: E402


def make_carrier(path: str, size="640x480", rate=24, seconds=4) -> None:
    subprocess.run([
        ffmpeg_io.ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", f"testsrc2=size={size}:rate={rate}",
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100",
        "-t", str(seconds), "-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p",
        "-c:a", "aac", path,
    ], check=True)


def test_dct_codec_all_modes():
    rng = np.random.default_rng(1)
    n = dct.block_count(480, 640)
    pos = bitstream.frame_permutation(b"k", 0, n)
    planes = [rng.integers(0, 256, (480, 640), dtype=np.uint8),
              np.zeros((480, 640), np.uint8), np.full((480, 640), 255, np.uint8)]
    for k in (1, 4, dct.MAX_BITS_PER_BLOCK):
        bits = rng.integers(0, 2, n * k - 7, dtype=np.uint8)  # partial last pair
        for plane in planes:
            out = dct.embed_bits(plane, bits, pos, 10.0, bits_per_block=k)
            got = dct.extract_bits(out, pos, k).ravel()[: len(bits)]
            assert (got == bits).all(), f"mode {k} failed"


def test_lsb_codec():
    rng = np.random.default_rng(2)
    n = ffmpeg_io.frame_size(64, 48)
    pos = bitstream.frame_permutation(b"k", 3, n)
    for b in (1, 2, lsb.MAX_BITS_PER_BYTE):
        frame = bytearray(rng.integers(0, 256, n, dtype=np.uint8).tobytes())
        original = np.frombuffer(bytes(frame), dtype=np.uint8)
        bits = rng.integers(0, 2, n * b - 5, dtype=np.uint8)
        lsb.embed_bits(frame, bits, pos, b)
        got = lsb.extract_bits(frame, pos, b).ravel()[: len(bits)]
        assert (got == bits).all()
        changed = np.frombuffer(bytes(frame), dtype=np.uint8)
        assert int(np.abs(changed.astype(int) - original.astype(int)).max()) < (1 << b)


def test_majority_vote_corrects_errors():
    bits = np.array([1, 0, 1, 1], dtype=np.uint8)
    rep = bitstream.repeat_bits(bits, 5)
    rep[0] ^= 1
    rep[6] ^= 1
    rep[7] ^= 1
    assert (bitstream.majority_vote(rep, 5) == bits).all()
    assert (bitstream.majority_vote(bitstream.repeat_bits(bits, 1), 1) == bits).all()


def _roundtrip_blob(path: str, data: bytes, password: str, chunk_size: int, open_pw: str = None) -> bytes:
    with open(path, "wb") as f:
        f.write(data)
    buf = io.BytesIO()
    pending = container.write_blob(crypto.iter_record(path), password, buf, chunk_size=chunk_size)
    container.finalize_blob(buf, pending, container.METHOD_DCT, 3, 5)
    assert len(buf.getvalue()) == pending.total_len == container.blob_length(crypto.record_length(path), chunk_size)
    hdr = container.parse_header(buf.getvalue()[: container.HEADER_LEN])
    assert (hdr.method, hdr.bits, hdr.repeat) == (container.METHOD_DCT, 3, 5)
    sink = io.BytesIO()
    writer = container.read_blob(buf, open_pw or password, lambda name, size: sink)
    assert writer.name == os.path.basename(path)
    return sink.getvalue()


def test_container_streaming_and_wrong_password():
    tmp = tempfile.mkdtemp(prefix="vstego_")
    path = os.path.join(tmp, "secret.txt")
    data = os.urandom(50_000) + b"compressible " * 5000
    assert _roundtrip_blob(path, data, "pw", chunk_size=4096) == data  # many chunks
    assert _roundtrip_blob(path, b"", "pw", chunk_size=4096) == b""      # empty file
    try:
        _roundtrip_blob(path, data, "pw", chunk_size=4096, open_pw="nope")
        raise AssertionError("expected WrongPasswordError")
    except crypto.WrongPasswordError:
        pass


def _video_roundtrip(tmp: str, carrier: str, size: int, label: str, **hide_kw) -> int:
    secret = os.path.join(tmp, f"secret_{label}.bin")
    out = os.path.join(tmp, f"stego_{label}.mkv")
    data = os.urandom(size)
    with open(secret, "wb") as f:
        f.write(data)
    t = time.time()
    res = stego.hide(carrier, secret, out, "hunter2", **hide_kw)
    out_size = os.path.getsize(out)
    print(f"[{label}] hide: {res.mode_label}, {res.frames_used}/{res.frames_total} frames, "
          f"{res.audio_note}, {out_size / 2**20:.1f} MB, {time.time() - t:.1f}s")
    assert ffmpeg_io.probe(out).has_audio
    t = time.time()
    ext = stego.extract(out, "hunter2", os.path.join(tmp, "out"))
    print(f"[{label}] extract: {ext.output} ({ext.mode_label}) in {time.time() - t:.1f}s")
    assert (ext.method, ext.bits, ext.repeat) == (res.method, res.bits, res.repeat)
    with open(ext.output, "rb") as f:
        assert f.read() == data
    return out_size


def test_video_roundtrip(tmp_path=None):
    tmp = str(tmp_path or tempfile.mkdtemp(prefix="vstego_"))
    carrier = os.path.join(tmp, "carrier.mp4")
    make_carrier(carrier)
    info = ffmpeg_io.probe(carrier)
    print(f"carrier {info.width}x{info.height} @ {float(info.fps):.2f} fps, {info.bitrate_kbps} kb/s, "
          f"audio={info.has_audio}; LSB capacity {stego.capacity_for(info, stego.METHOD_LSB, 4, 1).bytes_total} B, "
          f"DCT capacity {stego.capacity_for(info, stego.METHOD_DCT, 8, 1).bytes_total} B")

    sizes = {}
    sizes["lsb"] = _video_roundtrip(tmp, carrier, 200_000, "lsb")                       # auto LSB
    sizes["lsb4"] = _video_roundtrip(tmp, carrier, 1_500_000, "lsb4", bits=4, chunk_size=65536)
    sizes["dct"] = _video_roundtrip(tmp, carrier, 7_000, "dct", method=stego.METHOD_DCT)  # auto DCT
    sizes["dctmax"] = _video_roundtrip(tmp, carrier, 300_000, "dctmax", method=stego.METHOD_DCT,
                                       bits=8, repeat=1, chunk_size=65536)
    sizes["all"] = _video_roundtrip(tmp, carrier, 7_000, "all", lossless_all=True)
    print("sizes:", {k: f"{v / 2**20:.1f} MB" for k, v in sizes.items()})
    assert sizes["lsb"] < sizes["all"]

    stego_file = os.path.join(tmp, "stego_lsb.mkv")
    try:
        stego.extract(stego_file, "wrong-password", os.path.join(tmp, "out"))
        raise AssertionError("expected failure with wrong password")
    except (container.NotStegoError, crypto.WrongPasswordError) as exc:
        print("wrong password ->", exc)
    try:
        stego.extract(carrier, "hunter2", os.path.join(tmp, "out"))
        raise AssertionError("expected failure on clean carrier")
    except container.NotStegoError as exc:
        print("clean carrier ->", exc)


if __name__ == "__main__":
    test_dct_codec_all_modes()
    test_lsb_codec()
    test_majority_vote_corrects_errors()
    test_container_streaming_and_wrong_password()
    test_video_roundtrip()
    print("ALL TESTS PASSED")
