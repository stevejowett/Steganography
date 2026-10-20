"""Tkinter GUI with an encoder ("Hide") tab and a decoder ("Extract") tab."""

from __future__ import annotations

import os
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from . import __version__, bitstream, dct, ffmpeg_io, lsb, stego

VIDEO_TYPES = [("Video files", "*.mp4 *.mkv *.mov *.avi *.webm *.m4v *.mpg *.mpeg *.ts *.wmv"),
               ("All files", "*.*")]
METHOD_CHOICES = {"LSB (compact)": stego.METHOD_LSB, "DCT (robust)": stego.METHOD_DCT}
QUALITY_CHOICES = {"same bitrate as source": None, "CRF 18 (high)": 18, "CRF 20": 20,
                   "CRF 23 (normal)": 23, "CRF 26 (small)": 26}


def _fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:,.0f} {unit}" if unit == "B" else f"{n:,.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} GB"


class _Tab(ttk.Frame):
    """Shared scaffolding: path rows, password row, progress bar, log, worker thread."""

    def __init__(self, master: tk.Misc, app: "App"):
        super().__init__(master, padding=12)
        self.app = app
        self.columnconfigure(1, weight=1)
        self._row = 0
        self._queue: queue.Queue = queue.Queue()
        self._running = False

    def _add_path(self, label: str, browse) -> tk.StringVar:
        var = tk.StringVar()
        ttk.Label(self, text=label).grid(row=self._row, column=0, sticky="w", pady=4)
        ttk.Entry(self, textvariable=var).grid(row=self._row, column=1, sticky="ew", padx=6, pady=4)
        ttk.Button(self, text="Browse...", command=lambda: browse(var)).grid(row=self._row, column=2, pady=4)
        self._row += 1
        return var

    def _add_password(self, label: str) -> tk.StringVar:
        var = tk.StringVar()
        ttk.Label(self, text=label).grid(row=self._row, column=0, sticky="w", pady=4)
        entry = ttk.Entry(self, textvariable=var, show="*")
        entry.grid(row=self._row, column=1, sticky="ew", padx=6, pady=4)
        show = tk.BooleanVar(value=False)
        ttk.Checkbutton(self, text="Show", variable=show,
                        command=lambda: entry.config(show="" if show.get() else "*")
                        ).grid(row=self._row, column=2, sticky="w")
        self._row += 1
        return var

    def _add_info(self) -> tk.StringVar:
        var = tk.StringVar()
        ttk.Label(self, textvariable=var, foreground="#555", wraplength=640, justify="left"
                  ).grid(row=self._row, column=0, columnspan=3, sticky="w", pady=(2, 8))
        self._row += 1
        return var

    def _add_action(self, text: str, command) -> ttk.Button:
        btn = ttk.Button(self, text=text, command=command)
        btn.grid(row=self._row, column=0, columnspan=3, pady=(8, 4), sticky="ew")
        self._row += 1
        self.progress = ttk.Progressbar(self, mode="determinate", maximum=100)
        self.progress.grid(row=self._row, column=0, columnspan=3, sticky="ew", pady=2)
        self._row += 1
        self.status = tk.StringVar(value="Ready.")
        ttk.Label(self, textvariable=self.status).grid(row=self._row, column=0, columnspan=3, sticky="w")
        self._row += 1
        self.log = tk.Text(self, height=8, wrap="word", state="disabled", font=("Consolas", 9))
        self.log.grid(row=self._row, column=0, columnspan=3, sticky="nsew", pady=(6, 0))
        self.rowconfigure(self._row, weight=1)
        self._row += 1
        return btn

    # -- helpers used from the worker thread -------------------------------------------
    def _progress_cb(self, done: int, total: int, msg: str) -> None:
        self._queue.put(("progress", done, total, msg))

    def _log(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _run_worker(self, button: ttk.Button, target, on_done) -> None:
        if self._running:
            return
        self._running = True
        button.configure(state="disabled")
        self.progress["value"] = 0

        def work():
            try:
                result = target()
                self._queue.put(("done", result))
            except Exception as exc:  # surface everything to the GUI
                self._queue.put(("error", exc))

        threading.Thread(target=work, daemon=True).start()

        def poll():
            try:
                while True:
                    item = self._queue.get_nowait()
                    if item[0] == "progress":
                        _, done, total, msg = item
                        self.progress["value"] = 100 * done / total if total else 0
                        self.status.set(msg)
                    elif item[0] == "done":
                        self._running = False
                        button.configure(state="normal")
                        self.progress["value"] = 100
                        on_done(item[1])
                        return
                    else:
                        self._running = False
                        button.configure(state="normal")
                        self.status.set("Failed.")
                        self._log(f"ERROR: {item[1]}")
                        messagebox.showerror("Error", str(item[1]))
                        return
            except queue.Empty:
                pass
            self.after(100, poll)

        poll()


class HideTab(_Tab):
    def __init__(self, master, app):
        super().__init__(master, app)
        self.carrier = self._add_path("Carrier video:", self._pick_carrier)
        self.carrier_info = self._add_info()
        self.secret = self._add_path("Secret file:", self._pick_secret)
        self.secret_info = self._add_info()
        self.output = self._add_path("Output video:", self._pick_output)
        self.password = self._add_password("Password:")
        self.confirm = self._add_password("Confirm password:")

        # -- embedding settings ---------------------------------------------------------
        ttk.Label(self, text="Embedding:").grid(row=self._row, column=0, sticky="w", pady=4)
        frame = ttk.Frame(self)
        frame.grid(row=self._row, column=1, columnspan=2, sticky="ew", padx=6)
        self.method = tk.StringVar(value="LSB (compact)")
        cbm = ttk.Combobox(frame, textvariable=self.method, width=14, state="readonly",
                           values=list(METHOD_CHOICES))
        cbm.pack(side="left", padx=(0, 10))
        cbm.bind("<<ComboboxSelected>>", lambda _e: self._method_changed())
        ttk.Label(frame, text="bits").pack(side="left")
        self.bits = tk.StringVar(value="Auto")
        self.bits_box = ttk.Combobox(frame, textvariable=self.bits, width=5, state="readonly")
        self.bits_box.pack(side="left", padx=(2, 10))
        self.bits_box.bind("<<ComboboxSelected>>", lambda _e: self._update_fit())
        ttk.Label(frame, text="repetition").pack(side="left")
        self.repeat = tk.StringVar(value="Auto")
        cbr = ttk.Combobox(frame, textvariable=self.repeat, width=5, state="readonly",
                           values=["Auto"] + [str(r) for r in bitstream.REPEAT_OPTIONS])
        cbr.pack(side="left", padx=(2, 10))
        cbr.bind("<<ComboboxSelected>>", lambda _e: self._update_fit())
        ttk.Label(frame, text="DCT strength").pack(side="left")
        self.strength = tk.DoubleVar(value=dct.DEFAULT_STRENGTH)
        ttk.Spinbox(frame, from_=4, to=40, increment=1, textvariable=self.strength, width=4).pack(side="left", padx=(2, 0))
        self._row += 1

        # -- output size settings -------------------------------------------------------
        ttk.Label(self, text="Other frames:").grid(row=self._row, column=0, sticky="w", pady=4)
        frame2 = ttk.Frame(self)
        frame2.grid(row=self._row, column=1, columnspan=2, sticky="ew", padx=6)
        self.quality = tk.StringVar(value="same bitrate as source")
        cbq = ttk.Combobox(frame2, textvariable=self.quality, width=24, state="readonly",
                           values=list(QUALITY_CHOICES))
        cbq.pack(side="left", padx=(0, 10))
        self.lossless_all = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame2, text="all frames lossless (huge file)", variable=self.lossless_all,
                        command=self._update_fit).pack(side="left")
        self._row += 1
        self.fit_info = self._add_info()
        self._method_changed(initial=True)

        self.button = self._add_action("Hide file in video", self._start)
        self._info = None
        self._carrier_bytes = 0
        self._need = None

    def _method_changed(self, initial: bool = False):
        m = METHOD_CHOICES[self.method.get()]
        self.bits_box.configure(values=["Auto"] + [str(i) for i in range(1, stego.max_bits(m) + 1)])
        self.bits.set("Auto")
        if not initial:
            self._update_fit()
        if self._info is None if not initial else True:
            self.fit_info.set(
                "LSB stores 1-4 bits in every pixel byte of the data frames (1-2 bits are invisible) "
                "and costs about one coded bit per hidden bit. DCT hides 1-8 bits per 8x8 block in "
                "coefficient signs; it is far more robust but ~15x more expensive. Only data frames "
                "are stored losslessly; Auto keeps them under ~10% of the video.")

    # -- pickers -------------------------------------------------------------------------
    def _pick_carrier(self, var):
        path = filedialog.askopenfilename(title="Select carrier video", filetypes=VIDEO_TYPES)
        if not path:
            return
        var.set(path)
        if not self.output.get():
            base, _ = os.path.splitext(path)
            self.output.set(base + "_stego.mkv")
        self.carrier_info.set("Probing...")
        threading.Thread(target=self._probe_carrier, args=(path,), daemon=True).start()

    def _probe_carrier(self, path):
        try:
            info = ffmpeg_io.probe(path)
            size = os.path.getsize(path)
            lo = stego.capacity_for(info, stego.METHOD_LSB, 1, 1)
            hi = stego.capacity_for(info, stego.METHOD_LSB, lsb.MAX_BITS_PER_BYTE, 1)
            text = (f"{info.width}x{info.height}, {float(info.fps):.2f} fps, {info.duration:.1f}s "
                    f"(~{info.frame_estimate} frames), {_fmt_bytes(size)}, ~{info.bitrate_kbps} kb/s, "
                    f"audio: {'yes' if info.has_audio else 'no'}. LSB capacity {_fmt_bytes(lo.bytes_total)} "
                    f"(1 bit/byte) to {_fmt_bytes(hi.bytes_total)} ({lsb.MAX_BITS_PER_BYTE} bits/byte).")
        except Exception as exc:
            info, size = None, 0
            text = f"Could not read video: {exc}"
        self._queue_ui(lambda: (setattr(self, "_info", info), setattr(self, "_carrier_bytes", size),
                                self.carrier_info.set(text), self._update_fit()))

    def _pick_secret(self, var):
        path = filedialog.askopenfilename(title="Select file to hide")
        if not path:
            return
        var.set(path)
        self.secret_info.set("Measuring (compressing)...")
        threading.Thread(target=self._measure_secret, args=(path,), daemon=True).start()

    def _measure_secret(self, path):
        try:
            need = stego.payload_size(path)
            text = f"{_fmt_bytes(os.path.getsize(path))} on disk -> {_fmt_bytes(need)} after compression + encryption."
        except Exception as exc:
            need = None
            text = f"Could not read file: {exc}"
        self._queue_ui(lambda: (setattr(self, "_need", need), self.secret_info.set(text), self._update_fit()))

    def _queue_ui(self, fn):
        """Run ``fn`` on the Tk thread."""
        self.after(0, fn)

    def _settings(self) -> tuple[int, int, int]:
        m = METHOD_CHOICES[self.method.get()]
        b = stego.AUTO if self.bits.get() == "Auto" else int(self.bits.get())
        r = stego.AUTO if self.repeat.get() == "Auto" else int(self.repeat.get())
        return m, b, r

    def _update_fit(self):
        if self._info is None or self._need is None:
            return
        m, b, r = self._settings()
        cap = stego.choose_mode(self._info, self._need, m, b, r)
        if cap is None:
            best = stego.capacity_for(self._info, m, stego.max_bits(m), 1)
            self.fit_info.set(f"DOES NOT FIT with these settings (carrier maximum for "
                              f"{stego.METHOD_NAMES[m].upper()} is {_fmt_bytes(best.bytes_total)}). "
                              "Choose Auto, a longer/larger video, or a smaller file.")
            return
        frames = stego.frames_needed(cap, self._need)
        share = 100.0 * frames / max(cap.frames, 1)
        est = stego.estimate_output_bytes(self._info, cap, self._need, self._carrier_bytes,
                                          lossless_all=bool(self.lossless_all.get()))
        self.fit_info.set(f"Fits: {cap.label}; {frames} of ~{cap.frames} frames carry data "
                          f"({share:.0f}% lossless). Estimated output ~{_fmt_bytes(est)}"
                          + (" (all frames lossless)." if self.lossless_all.get() else
                             ". Fewer data frames = smaller file; more bits = more visible."))

    def _pick_output(self, var):
        path = filedialog.asksaveasfilename(title="Save stego video as", defaultextension=".mkv",
                                            filetypes=[("Matroska", "*.mkv"), ("MP4", "*.mp4")])
        if path:
            var.set(path)

    # -- action --------------------------------------------------------------------------
    def _start(self):
        carrier, secret, output = self.carrier.get(), self.secret.get(), self.output.get()
        pw = self.password.get()
        if not (carrier and secret and output):
            messagebox.showwarning("Missing input", "Select a carrier video, a secret file and an output path.")
            return
        if not pw:
            messagebox.showwarning("Password", "Enter a password.")
            return
        if pw != self.confirm.get():
            messagebox.showwarning("Password", "Passwords do not match.")
            return
        if os.path.abspath(output) == os.path.abspath(carrier):
            messagebox.showwarning("Output", "Output must differ from the carrier.")
            return
        m, b, r = self._settings()
        strength = float(self.strength.get())
        lossless_all = bool(self.lossless_all.get())
        crf = QUALITY_CHOICES[self.quality.get()]
        self._log(f"Hiding {os.path.basename(secret)} in {os.path.basename(carrier)} ...")

        def on_done(res: stego.HideResult):
            self.status.set("Done.")
            try:
                size = _fmt_bytes(os.path.getsize(res.output))
            except OSError:
                size = "?"
            self._log(f"Written: {res.output} ({size})")
            self._log(f"Payload {res.payload_bytes:,} bytes, {res.mode_label}, in {res.frames_used} of "
                      f"{res.frames_total} frames ({'all frames' if res.lossless_all else 'data frames only'} "
                      f"lossless); {res.audio_note}.")
            messagebox.showinfo("Done", f"Stego video written to:\n{res.output}\n({size})")

        self._run_worker(self.button,
                         lambda: stego.hide(carrier, secret, output, pw, method=m, bits=b, repeat=r,
                                            strength=strength, lossless_all=lossless_all, crf=crf,
                                            progress=self._progress_cb),
                         on_done)


class ExtractTab(_Tab):
    def __init__(self, master, app):
        super().__init__(master, app)
        self.video = self._add_path("Stego video:", self._pick_video)
        self.password = self._add_password("Password:")
        self.outdir = self._add_path("Output folder:", self._pick_dir)
        self.button = self._add_action("Extract hidden file", self._start)

    def _pick_video(self, var):
        path = filedialog.askopenfilename(title="Select stego video", filetypes=VIDEO_TYPES)
        if path:
            var.set(path)
            if not self.outdir.get():
                self.outdir.set(os.path.dirname(path))

    def _pick_dir(self, var):
        path = filedialog.askdirectory(title="Select output folder")
        if path:
            var.set(path)

    def _start(self):
        video, pw, outdir = self.video.get(), self.password.get(), self.outdir.get()
        if not video or not outdir:
            messagebox.showwarning("Missing input", "Select a stego video and an output folder.")
            return
        if not pw:
            messagebox.showwarning("Password", "Enter the password.")
            return
        self._log(f"Extracting from {os.path.basename(video)} ...")

        def on_done(res: stego.ExtractResult):
            self.status.set("Done.")
            self._log(f"Recovered: {res.output} ({res.size:,} bytes; {res.mode_label})")
            self._log(f"SHA-256 verified: {res.sha256_hex}")
            messagebox.showinfo("Done", f"Hidden file recovered and verified:\n{res.output}")

        self._run_worker(self.button,
                         lambda: stego.extract(video, pw, outdir, progress=self._progress_cb),
                         on_done)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(f"Video Steganography v{__version__}")
        self.minsize(760, 620)
        try:
            ttk.Style(self).theme_use("vista")
        except tk.TclError:
            pass
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True)
        nb.add(HideTab(nb, self), text="  Hide (encode)  ")
        nb.add(ExtractTab(nb, self), text="  Extract (decode)  ")


def main() -> None:
    try:
        ffmpeg_io.ffmpeg_exe()
    except Exception as exc:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("ffmpeg missing", str(exc))
        return
    App().mainloop()


if __name__ == "__main__":
    main()
