"""Command-line interface.

    python -m videostego hide    carrier.mp4 secret.pdf out.mkv [-p PASSWORD] [-m lsb|dct] [-b N] [-r N]
    python -m videostego extract out.mkv  [-o DIR] [-p PASSWORD]
    python -m videostego info    carrier.mp4 [--secret FILE]
    python -m videostego gui
"""

from __future__ import annotations

import argparse
import getpass
import os
import sys

from . import bitstream, dct, ffmpeg_io, lsb, stego


def _password(args, confirm: bool) -> str:
    if args.password:
        return args.password
    pw = getpass.getpass("Password: ")
    if confirm and getpass.getpass("Confirm password: ") != pw:
        sys.exit("passwords do not match")
    return pw


def _progress(done: int, total: int, msg: str) -> None:
    pct = int(100 * done / total) if total else 0
    sys.stderr.write(f"\r[{pct:3d}%] {msg:<70}")
    sys.stderr.flush()


def _mb(n: float) -> str:
    return f"{n / 2**20:,.1f} MB"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="videostego",
                                 description="Hide encrypted files inside video.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("hide", help="embed a secret file into a carrier video")
    p.add_argument("carrier")
    p.add_argument("secret")
    p.add_argument("output", help="output video (.mkv recommended, .mp4 also works)")
    p.add_argument("-p", "--password")
    p.add_argument("-m", "--method", choices=("lsb", "dct"), default="lsb",
                   help="lsb = compact bit-plane embedding (default); dct = DCT-domain differential embedding")
    p.add_argument("-b", "--bits", type=int, default=stego.AUTO,
                   help=f"bits per byte (lsb, 1..{lsb.MAX_BITS_PER_BYTE}) or per 8x8 block "
                        f"(dct, 1..{dct.MAX_BITS_PER_BLOCK}); 0 = auto (default)")
    p.add_argument("-r", "--repeat", type=int, default=stego.AUTO, choices=(stego.AUTO,) + bitstream.REPEAT_OPTIONS,
                   help="bit repetition factor 5, 3 or 1; 0 = auto (default: 1 for lsb, 5 for dct if it fits)")
    p.add_argument("-s", "--strength", type=float, default=dct.DEFAULT_STRENGTH,
                   help=f"dct embedding strength (default {dct.DEFAULT_STRENGTH:g})")
    p.add_argument("--crf", type=int, default=None,
                   help="x264 CRF for frames that carry no data (default: match the source bitrate)")
    p.add_argument("--all-lossless", action="store_true",
                   help="encode every frame losslessly (very large output)")
    p.add_argument("--preset", default="fast", help="x264 preset (ultrafast..veryslow)")

    p = sub.add_parser("extract", help="recover the hidden file from a stego video")
    p.add_argument("video")
    p.add_argument("-o", "--output-dir", default=".")
    p.add_argument("-p", "--password")
    p.add_argument("--overwrite", action="store_true")

    p = sub.add_parser("info", help="show carrier details and capacity")
    p.add_argument("video")
    p.add_argument("--secret", help="also report the payload size and the auto mode for this file")

    sub.add_parser("gui", help="launch the graphical interface")

    args = ap.parse_args(argv)

    if args.cmd == "gui":
        from .gui import main as gui_main
        gui_main()
        return 0

    if args.cmd == "info":
        info = ffmpeg_io.probe(args.video)
        print(f"Resolution : {info.width}x{info.height}  ({info.codec}, {info.pix_fmt})")
        print(f"Frame rate : {float(info.fps):.3f} fps   duration {info.duration:.2f}s  (~{info.frame_estimate} frames)")
        print(f"Audio      : {'yes' if info.has_audio else 'no'}   bitrate ~{info.bitrate_kbps} kb/s")
        print("Capacity   : LSB  " + "  ".join(
            f"{b} bit/byte: {_mb(stego.capacity_for(info, stego.METHOD_LSB, b, 1).bytes_total)}"
            for b in range(1, lsb.MAX_BITS_PER_BYTE + 1)))
        print("             DCT  " + "  ".join(
            f"{b} bit/block: {_mb(stego.capacity_for(info, stego.METHOD_DCT, b, 1).bytes_total)}"
            for b in (1, 4, dct.MAX_BITS_PER_BLOCK)) + "  (1x repetition)")
        if args.secret:
            need = stego.payload_size(args.secret)
            carrier_bytes = os.path.getsize(args.video)
            print(f"Payload    : {need:,} bytes ({_mb(need)})")
            for method in (stego.METHOD_LSB, stego.METHOD_DCT):
                cap = stego.choose_mode(info, need, method)
                if cap:
                    n = stego.frames_needed(cap, need)
                    est = stego.estimate_output_bytes(info, cap, need, carrier_bytes)
                    print(f"  {stego.METHOD_NAMES[method].upper():3} auto : {cap.label} -> "
                          f"{n} of ~{cap.frames} frames lossless ({100 * n / cap.frames:.0f}%), "
                          f"output ~{_mb(est)}")
                else:
                    print(f"  {stego.METHOD_NAMES[method].upper():3} auto : payload does not fit")
        return 0

    try:
        if args.cmd == "hide":
            pw = _password(args, confirm=True)
            res = stego.hide(args.carrier, args.secret, args.output, pw,
                             method=stego.METHODS_BY_NAME[args.method], bits=args.bits,
                             repeat=args.repeat, strength=args.strength,
                             lossless_all=args.all_lossless, crf=args.crf,
                             preset=args.preset, progress=_progress)
            sys.stderr.write("\n")
            print(f"Written {res.output}  ({_mb(os.path.getsize(res.output))})")
            print(f"Payload {res.payload_bytes:,} bytes, {res.mode_label}, "
                  f"in {res.frames_used}/{res.frames_total} frames "
                  f"({'all frames' if res.lossless_all else 'data frames only'} lossless); {res.audio_note}.")
        else:
            pw = _password(args, confirm=False)
            res = stego.extract(args.video, pw, args.output_dir, progress=_progress,
                                overwrite=args.overwrite)
            sys.stderr.write("\n")
            print(f"Extracted {res.output}  ({res.size:,} bytes; {res.mode_label})")
            print(f"SHA-256 {res.sha256_hex}  verified OK")
    except Exception as exc:  # user-facing errors
        sys.stderr.write("\n")
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
