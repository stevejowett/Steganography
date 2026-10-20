# Video Steganography (DCT-domain, AES-256-GCM)

Hide any file inside a video. The secret is compressed, encrypted with a password, and written
into the luminance (Y) channel of the video in the DCT domain. The output is a losslessly
encoded H.264 video (x264 QP 0) with the original audio, so the hidden data survives
intact and can be recovered later with the password.

## Features

- Pick any **carrier video** (mp4, mkv, mov, avi, webm, ...) and any **secret file**.
- **Password-based AES-256-GCM** encryption, key derived with scrypt (N=2^15, r=8, p=1).
  Large files are encrypted in 1 MiB chunks, each chunk with its own nonce and tag.
- **zlib compression** of the secret before encryption.
- Two embedding methods, detected automatically on extraction:
  - **LSB (compact, default)**: 1-4 bits in the low bit-planes of every pixel byte (Y, U and
    V) of the data frames. 1-2 bits are invisible. Costs about one coded bit per hidden bit,
    so the output is close to carrier + payload.
  - **DCT (robust)**: 1-8 bits per 8x8 luminance block, each stored in the sign of the
    difference of two mid-frequency coefficients such as (3,2) and (2,3), with optional
    5x/3x repetition and majority vote. A verify-and-boost pass guarantees that rounding to
    8-bit never flips a bit. About 15x more expensive per hidden bit than LSB.
- **Auto** picks bits (and repetition for DCT) so that the lossless data frames stay under
  about 10% of the video, preferring the least visible settings that achieve it.
- Keyed, per-frame **pseudo-random block permutation** derived from the password: without the
  password the bit positions look random and the header cannot even be located.
- **Lossless H.264 (QP 0)** for every frame that carries data, so extraction is bit-exact.
  Frames that carry no data are re-encoded at the carrier's own bitrate (or a chosen CRF),
  so that part of the output stays about as large as the carrier. All-lossless is still
  available as an option.
- **Original audio preserved**: stream-copied when the container allows it, re-encoded to AAC
  otherwise, dropped only as a last resort (the result is reported).
- **Integrity verification** at three levels: CRC-32 on the header and on the ciphertext,
  the AES-GCM authentication tag of every chunk, and a SHA-256 of the original file stored
  inside the encrypted record.
- **Streaming end to end**: memory use does not grow with the file size. The payload is
  staged in a temporary file next to the output and frames are piped through ffmpeg.
- **GUI** (tkinter, two tabs: Hide / Extract) and a **CLI**.
- Everything, including ffmpeg, lives in the project `venv`.

## Setup (Windows)

Requires Python 3.10 or newer. Tkinter ships with the python.org installer.

```bat
setup_venv.bat
```

or in PowerShell:

```powershell
.\setup_venv.ps1
```

This creates `venv\` and installs `numpy`, `cryptography` and `imageio-ffmpeg` (which bundles a
static `ffmpeg` binary, so nothing needs to be on your PATH).

Manual equivalent:

```bat
py -3 -m venv venv
venv\Scripts\python -m pip install -r requirements.txt
```

## Usage

### GUI

```bat
run_gui.bat
```

or

```bat
venv\Scripts\python -m videostego gui
```

**Hide tab**: choose the carrier video, the secret file, an output path (`.mkv` recommended,
`.mp4` also works), type and confirm a password, press *Hide file in video*. The tab shows the
carrier capacity, the payload size after compression, whether it fits with the selected
method / bits / repetition, how many frames will be lossless, and an estimated output size.

**Extract tab**: choose the stego video, type the password, choose an output folder, press
*Extract hidden file*. The embedding mode is detected automatically. The file is restored with
its original name and verified against the embedded SHA-256.

### CLI

```bat
venv\Scripts\python -m videostego info    carrier.mp4 --secret secret.pdf
venv\Scripts\python -m videostego hide    carrier.mp4 secret.pdf out.mkv
venv\Scripts\python -m videostego hide    carrier.mp4 big.zip out.mkv -b 4
venv\Scripts\python -m videostego hide    carrier.mp4 secret.pdf out.mkv -m dct -b 1 -r 5
venv\Scripts\python -m videostego extract out.mkv -o recovered\
```

The password is prompted for interactively, or passed with `-p`. Options for `hide`:

| Option | Meaning |
| ------ | ------- |
| `-m, --method {lsb,dct}` | embedding method (default `lsb`) |
| `-b, --bits N` | bits per byte (lsb, 1..4) or per 8x8 block (dct, 1..8); 0 = auto (default) |
| `-r, --repeat {0,5,3,1}` | bit repetition factor; 0 = auto (default: 1 for lsb) |
| `--crf N` | x264 CRF for frames without data (default: match the carrier's bitrate) |
| `--all-lossless` | encode every frame losslessly (very large output) |
| `-s, --strength S` | DCT embedding strength (default 10) |
| `--preset P` | x264 preset, `ultrafast`..`veryslow` (default `fast`) |

### Tests

```bat
venv\Scripts\python tests\test_roundtrip.py
```

The tests cover both codecs (DCT in every mode including saturated black/white frames, LSB
at 1-4 bits), the majority vote, the chunked container with a wrong password, and five video
round trips on a synthesized carrier with audio: LSB auto, LSB 4 bits with multi-chunk
encryption, DCT auto, DCT maximum capacity, and all-lossless. They also check the
wrong-password and clean-carrier error paths.

## How it works

```text
secret file
  -> record: filename | size | zlib stream | SHA-256          (streamed, never fully in RAM)
  -> 1 MiB chunks, each AES-256-GCM sealed (nonce = base || index, index + is-last in AAD)
  -> blob: magic | version | salt | nonce base | chunk size | mode | length | CRC-32
           | chunk_0 .. chunk_n | CRC-32                       (temp file next to the output)
  -> every bit repeated r times (DCT: 5, 3 or 1; LSB: 1)
  -> LSB: per frame, bits written into the low bit-planes of all frame bytes, in an order
          given by a password-keyed permutation
     DCT: per frame, bits placed in 8x8 Y blocks chosen by a password-keyed permutation;
          per block, for each of k coefficient pairs (a, b): DCT, push C(a) - C(b) to >= +S
          (bit 1) or <= -S (bit 0), IDCT, round to 8-bit, verify and boost if needed
  -> data frames: libx264 -qp 0 (lossless); remaining frames: libx264 at the carrier bitrate
     (two Annex-B streams joined into one H.264 stream), remux with the carrier's audio
```

Extraction replays the permutations for the first frame, reads both the LSB planes and the
DCT coefficient signs, and tries each (method, repetition, bits) combination until a header
with a valid magic and CRC-32 appears. It then majority-votes the stream incrementally
into a temporary blob file, stops reading the video once the payload is complete, checks the
CRC-32s, decrypts chunk by chunk (a wrong password fails on the first chunk), decompresses
straight to the output file, and compares the SHA-256.

### Capacity per frame

| Resolution | LSB 1 bit/byte | LSB 4 bits/byte | DCT 1 bit/block, 5x | DCT 8 bits/block, 1x |
| ---------: | -------------: | --------------: | ------------------: | -------------------: |
| 640x480    | 56 KB          | 225 KB          | 120 B               | 4.8 KB               |
| 1280x720   | 169 KB         | 675 KB          | 360 B               | 14.4 KB              |
| 1920x1080  | 380 KB         | 1.5 MB          | 810 B               | 32.4 KB              |
| 3840x2160  | 1.5 MB         | 6.1 MB          | 3.2 KB              | 129.6 KB             |

Multiply by the frame count. `info` prints the exact numbers for a given carrier. Only as
many frames as needed carry data, always the leading frames of the video. The rest are
passed through unchanged and re-encoded at the carrier's bitrate.

### Output size

Data frames are stored losslessly. A lossless 720p frame of natural video costs about
250 KB, a 1080p one about 600 KB, plus the embedded bits themselves (about one coded bit
per hidden bit for LSB, about 15 for DCT). Roughly:

```text
output ~= data_frames x (lossless frame cost + embedded bits)  +  carrier size x (1 - data share)
```

For a 720p, 25 fps, 5.7-minute, 12 MB carrier and a 9.7 MB payload:

| method | data frames | lossless part | output (approx.) |
| ------ | ----------: | ------------: | ---------------: |
| DCT 8 bits/block, 1x | 674 (8%) | ~160 MB | ~175 MB |
| LSB 1 bit/byte | 58 (0.7%) | ~24 MB | ~36 MB |
| LSB 2 bits/byte | 29 (0.3%) | ~17 MB | ~29 MB |
| LSB 4 bits/byte | 15 (0.2%) | ~14 MB | ~26 MB |

The floor is carrier + payload (about 22 MB here); LSB gets within 20-60% of it. For a
300 MB payload in a 500 MB 1080p carrier expect roughly 1 GB (LSB 2 bits) instead of 3 GB
(DCT). More bits per byte shrink the file further but become visible above 2.

### Things to know

- **The payload always adds to the size.** Encrypted data is incompressible, so the output
  can never be smaller than carrier + payload. LSB mode is the closest to that bound.
- **Do not re-encode the stego video** (YouTube, WhatsApp, HandBrake, ...). Lossy compression
  would destroy the payload. Copy the file as-is.
- The data frames are encoded as 4:2:0 lossless (H.264 High 4:4:4 Predictive profile) and
  the stream switches to the regular High profile afterwards. ffmpeg-based players (VLC, mpv,
  ffplay) handle this; some hardware decoders do not.
- Visibility: LSB at 1-2 bits/byte is invisible, 3-4 bits add faint noise. DCT at 1 bit/block
  and strength 8-14 is subtle; 8 bits/block adds visible texture. Repetition never affects
  visibility, only capacity.
- Robustness: both methods rely on the data frames being lossless. DCT additionally tolerates
  small pixel errors (repetition + sign margins); LSB does not. Neither survives re-encoding.
- Hiding needs free disk space for the encrypted payload (about the compressed file size) plus
  the lossless video, both next to the output path. Extracting needs the same for the payload.
- The per-frame permutation seed depends on the password only, not on the random salt
  (the salt is inside the header that the permutation protects). The confidentiality of
  the content rests on AES-256-GCM with a scrypt-derived key; the permutation adds
  deniability, not encryption.
- Odd video dimensions are trimmed to even values (required for 4:2:0).

## Project layout

```text
videostego/
  crypto.py       scrypt KDF, chunked AES-256-GCM, streaming zlib record with SHA-256
  container.py    embedded blob format with magic, CRC-32s, header parsing, blob I/O
  dct.py          8x8 DCT, differential embedding (1..8 pairs), verify-and-boost, extraction
  lsb.py          compact bit-plane embedding (1..4 bits per frame byte)
  bitstream.py    repetition, majority vote, keyed per-frame permutation
  ffmpeg_io.py    probe, raw yuv420p frame reader, lossless + lossy segmented x264 writer, audio mux
  stego.py        hide() / extract() streaming pipelines, capacity and mode selection
  gui.py          tkinter encoder / decoder
  cli.py          command-line interface (python -m videostego ...)
tests/test_roundtrip.py
requirements.txt, setup_venv.bat, setup_venv.ps1, run_gui.bat
```
