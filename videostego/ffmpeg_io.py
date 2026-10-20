"""Thin wrapper around the ffmpeg binary (bundled by imageio-ffmpeg).

Frames are exchanged as raw yuv420p: the Y plane is the first width*height
bytes of every frame, so the luminance channel can be edited in place without
any colour-space conversion or rounding.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from fractions import Fraction
from typing import Iterator, Optional

_CREATION_FLAGS = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
DEFAULT_CRF = 20


def ffmpeg_exe() -> str:
    """Locate ffmpeg: the imageio-ffmpeg bundle first, then PATH."""
    try:
        import imageio_ffmpeg  # type: ignore

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # pragma: no cover - fallback only
        found = shutil.which("ffmpeg")
        if not found:
            raise RuntimeError("ffmpeg not found: install imageio-ffmpeg or put ffmpeg on PATH")
        return found


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: Fraction
    duration: float
    has_audio: bool
    codec: str
    pix_fmt: str
    bitrate_kbps: int = 0  # overall bitrate of the source file (0 if unknown)

    @property
    def frame_estimate(self) -> int:
        return max(1, int(round(self.duration * float(self.fps))))

    @property
    def even_width(self) -> int:
        return self.width - (self.width % 2)

    @property
    def even_height(self) -> int:
        return self.height - (self.height % 2)


def _parse_rate(text: str) -> Fraction:
    text = text.strip()
    mult = 1
    if text.endswith("k"):
        mult = 1000
        text = text[:-1]
    return Fraction(text).limit_denominator(100000) * mult


def probe(path: str) -> VideoInfo:
    """Parse ``ffmpeg -i`` output (no ffprobe is bundled with imageio-ffmpeg)."""
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    proc = subprocess.run(
        [ffmpeg_exe(), "-hide_banner", "-i", path],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=_CREATION_FLAGS,
    )
    out = proc.stderr
    video_line = None
    has_audio = False
    for line in out.splitlines():
        if "Stream #" in line and "Video:" in line and video_line is None:
            video_line = line
        if "Stream #" in line and "Audio:" in line:
            has_audio = True
    if video_line is None:
        raise ValueError(f"no video stream found in {path}")

    m = re.search(r",\s*(\d{2,5})x(\d{2,5})[\s,\[]", video_line)
    if not m:
        raise ValueError(f"could not parse resolution from: {video_line}")
    width, height = int(m.group(1)), int(m.group(2))

    fps = None
    m = re.search(r"([\d.]+k?)\s*fps", video_line)
    if m:
        fps = _parse_rate(m.group(1))
    else:
        m = re.search(r"([\d.]+k?)\s*tbr", video_line)
        if m:
            fps = _parse_rate(m.group(1))
    if not fps or fps <= 0:
        fps = Fraction(30)

    duration = 0.0
    m = re.search(r"Duration:\s*(\d+):(\d+):([\d.]+)", out)
    if m:
        duration = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))

    m = re.search(r"Video:\s*(\w+)", video_line)
    codec = m.group(1) if m else "?"
    m = re.search(r"Video:[^,]*,\s*([\w]+)", video_line)
    pix_fmt = m.group(1) if m else "?"
    bitrate = 0
    m = re.search(r"bitrate:\s*(\d+)\s*kb/s", out)
    if m:
        bitrate = int(m.group(1))
    elif duration > 0:
        bitrate = int(os.path.getsize(path) * 8 / duration / 1000)
    return VideoInfo(width, height, fps, duration, has_audio, codec, pix_fmt, bitrate)


def frame_size(width: int, height: int) -> int:
    return width * height * 3 // 2  # yuv420p


def read_frames(path: str, width: int, height: int) -> Iterator[bytearray]:
    """Yield raw yuv420p frames (scaled to even dimensions if necessary)."""
    cmd = [
        ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-nostdin",
        "-noautorotate",  # keep decoded dimensions identical to the probed ones
        "-i", path,
        "-an", "-sn", "-dn",
        "-vf", f"scale={width}:{height}:flags=bicubic",
        "-f", "rawvideo", "-pix_fmt", "yuv420p",
        "pipe:1",
    ]
    size = frame_size(width, height)
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=size * 4,
        creationflags=_CREATION_FLAGS,
    )
    assert proc.stdout is not None
    try:
        while True:
            buf = bytearray()
            while len(buf) < size:
                chunk = proc.stdout.read(size - len(buf))
                if not chunk:
                    break
                buf.extend(chunk)
            if len(buf) < size:
                break
            yield buf
    finally:
        try:
            proc.stdout.close()
        except Exception:
            pass
        if proc.poll() is None:
            proc.kill()
        proc.wait()


class _X264Proc:
    """One ffmpeg process: raw yuv420p frames on stdin -> raw Annex-B H.264 file."""

    def __init__(self, path: str, width: int, height: int, fps: Fraction, video_args: list[str]):
        self.path = path
        self._log = path + ".log"
        cmd = [
            ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", f"{width}x{height}",
            "-r", f"{fps.numerator}/{fps.denominator}",
            "-i", "pipe:0",
            "-c:v", "libx264", *video_args, "-pix_fmt", "yuv420p",
            "-an", "-f", "h264",
            path,
        ]
        self._logf = open(self._log, "wb")
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self._logf,
            creationflags=_CREATION_FLAGS,
        )

    def write(self, frame: bytes | bytearray) -> None:
        assert self.proc.stdin is not None
        try:
            self.proc.stdin.write(frame)
        except (BrokenPipeError, OSError) as exc:
            raise RuntimeError(f"ffmpeg encoder died: {self._read_log()}") from exc

    def close(self) -> None:
        if self.proc.stdin:
            try:
                self.proc.stdin.close()
            except Exception:
                pass
        rc = self.proc.wait()
        self._logf.close()
        log = self._read_log()
        _remove(self._log)
        if rc != 0:
            raise RuntimeError(f"ffmpeg encoder failed (exit {rc}): {log}")

    def abort(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait()
        self._logf.close()
        _remove(self._log)
        _remove(self.path)

    def _read_log(self) -> str:
        try:
            with open(self._log, "rb") as f:
                return f.read().decode("utf-8", errors="replace").strip()[-2000:]
        except OSError:
            return ""


def _remove(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


class SegmentedWriter:
    """Encode the first ``lossless_frames`` frames with x264 lossless (QP 0) and
    every later frame with a normal lossy CRF, then join both Annex-B streams
    into one raw ``.h264`` file.

    Hidden data lives only in the leading frames, so only those must survive
    bit-exactly; encoding the remainder lossily shrinks the output by an order
    of magnitude. ``lossless_frames=None`` keeps the whole video lossless.
    """

    def __init__(self, path: str, width: int, height: int, fps: Fraction,
                 lossless_frames: Optional[int], crf: Optional[int] = None,
                 bitrate_kbps: int = 0, preset: str = "fast"):
        self.path = path
        self._dims = (width, height, fps)
        self._lossless_frames = lossless_frames
        if crf is not None:
            quality = ["-crf", str(int(crf))]
        elif bitrate_kbps > 0:
            # Match the source bitrate so the pass-through part stays about as big as the carrier.
            br = max(int(bitrate_kbps), 100)
            quality = ["-b:v", f"{br}k", "-maxrate", f"{2 * br}k", "-bufsize", f"{4 * br}k"]
        else:
            quality = ["-crf", str(DEFAULT_CRF)]
        self._lossy_args = quality + ["-preset", preset]
        self._parts = [path + ".lossless.h264", path + ".lossy.h264"]
        self._count = 0
        self._stage = 0
        self._active: Optional[_X264Proc] = None
        if lossless_frames is None or lossless_frames > 0:
            self._active = _X264Proc(self._parts[0], width, height, fps, ["-qp", "0", "-preset", preset])
        else:
            self._stage = 1
            self._active = _X264Proc(self._parts[1], width, height, fps, self._lossy_args)

    @property
    def frames_written(self) -> int:
        return self._count

    def write(self, frame: bytes | bytearray) -> None:
        assert self._active is not None
        if (self._stage == 0 and self._lossless_frames is not None
                and self._count >= self._lossless_frames):
            self._active.close()
            w, h, fps = self._dims
            self._active = _X264Proc(self._parts[1], w, h, fps, self._lossy_args)
            self._stage = 1
        self._active.write(frame)
        self._count += 1

    def close(self) -> str:
        """Finish encoding and return the path of the joined raw H.264 stream."""
        assert self._active is not None
        self._active.close()
        self._active = None
        with open(self.path, "wb") as out:
            for part in self._parts:
                if os.path.isfile(part):
                    with open(part, "rb") as src:
                        shutil.copyfileobj(src, out, 1 << 20)
                    _remove(part)
        return self.path

    def abort(self) -> None:
        if self._active is not None:
            self._active.abort()
            self._active = None
        for part in self._parts:
            _remove(part)
        _remove(self.path)


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", creationflags=_CREATION_FLAGS)


def mux_with_audio(video_path: str, audio_source: str, output: str,
                   fps: Optional[Fraction] = None) -> str:
    """Copy the encoded video stream into ``output`` and attach the carrier's audio.

    ``video_path`` may be a raw ``.h264`` stream (then ``fps`` is required to
    time-stamp it) or any container. Returns a short note about the audio.
    """
    in_args: list[str] = []
    if video_path.lower().endswith(".h264"):
        if fps is None:
            raise ValueError("fps is required for a raw .h264 input")
        in_args = ["-fflags", "+genpts", "-framerate", f"{fps.numerator}/{fps.denominator}"]
    base = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
            *in_args, "-i", video_path, "-i", audio_source,
            "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "copy"]
    attempts = [
        (base + ["-c:a", "copy", output], "audio copied losslessly"),
        (base + ["-c:a", "aac", "-b:a", "192k", output], "audio re-encoded to AAC"),
    ]
    errors = []
    for cmd, note in attempts:
        res = _run(cmd)
        if res.returncode == 0 and os.path.isfile(output):
            return note
        errors.append(res.stderr.strip())
    # Last resort: video only.
    res = _run([ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
                *in_args, "-i", video_path, "-map", "0:v:0", "-c:v", "copy", "-an", output])
    if res.returncode != 0:
        raise RuntimeError("muxing failed: " + " | ".join(errors + [res.stderr.strip()]))
    return "audio could not be preserved (video written without audio)"
