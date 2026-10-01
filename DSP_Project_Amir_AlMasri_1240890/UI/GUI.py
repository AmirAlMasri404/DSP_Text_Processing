"""Dark desktop interface for the DSP project. Run this file to start."""
import csv
import sys
from pathlib import Path

# Support direct execution as well as python -m UI.GUI.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from concurrent.futures import ThreadPoolExecutor
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import sounddevice as sd
from scipy.io import wavfile
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from UI.app_engine import Engine
from UI.plots import FREQUENCIES, make_plots

BG, CARD, FIELD = "#0b111c", "#141e2d", "#0c1523"
TEXT, MUTED, BLUE = "#e7f0fc", "#91a7c2", "#58c8ff"


def configure_dpi():
    """Establish Windows DPI handling before Tk or Matplotlib creates windows."""
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)


class DSPInterface:
    def __init__(self, root, engine=None):
        self.root, self.engine = root, engine or Engine()
        self.root.tk.call("tk", "scaling", 1.333333)
        self.root.geometry("1100x850")
        self.root.minsize(1000, 750)
        self.result, self.future = None, None
        self.records, self.closed = [], False
        self.expected = tk.StringVar(value="")
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.input_mode = tk.StringVar(value="Text")
        self.decoder_mode = tk.StringVar(value="Both")
        self.value = tk.StringVar(value="Enter Your String")
        self.status = tk.StringVar(value="Ready. Enter a message or select an encoded WAV.")
        self.metrics = tk.StringVar(value="No message processed yet")
        self.segment_number = tk.IntVar(value=1)
        self.filters = {f: tk.BooleanVar(value=False) for f in FREQUENCIES}
        self.filter_views = {v: tk.BooleanVar(value=False) for v in ["Time", "Spectrum", "Pole-zero"]}
        self.message_views = {v: tk.BooleanVar(value=False) for v in
                              ["Input time", "Input spectrum", "Recovered time", "Recovered spectrum"]}
        self.apply_style()
        self.build()
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.bind("<Control-Return>", lambda event: self.run())

    def apply_style(self):
        self.root.title("Text & Tone Lab | DSP")
        self.root.geometry("1120x850")
        self.root.minsize(980, 850)
        self.root.configure(bg=BG)
        self.root.option_add("*Font", "{Segoe UI} 10")
        s = ttk.Style()
        s.theme_use("clam")
        s.configure("TFrame", background=BG)
        s.configure("Card.TFrame", background=CARD)
        s.configure("TLabel", background=BG, foreground=TEXT)
        s.configure("Muted.TLabel", foreground=MUTED)
        s.configure("Card.TLabel", background=CARD, foreground=MUTED)
        for kind in ["TRadiobutton", "TCheckbutton"]:
            s.configure(kind, background=CARD, foreground=TEXT, padding=4)
            s.map(kind, background=[("active", CARD)],
                  indicatorbackground=[("selected", BLUE), ("!selected", FIELD)],
                  indicatorforeground=[("selected", BG)])
        s.configure("TEntry", fieldbackground=FIELD, foreground=TEXT, insertcolor=TEXT, padding=10)
        s.configure("TSpinbox", fieldbackground=FIELD, foreground=TEXT, arrowcolor=BLUE, padding=5)
        s.configure("TButton", background=BLUE, foreground=BG, padding=(14, 9),
                    borderwidth=0, font=("Segoe UI", 10, "bold"))
        s.map("TButton", background=[("disabled", "#253d51"), ("active", "#91dcff")],
              foreground=[("disabled", "#8195ac")])
        s.configure("Secondary.TButton", background="#22354b", foreground="#b7e9ff")
        s.map("Secondary.TButton", background=[("active", "#304b66")])
        s.configure("Treeview", background=FIELD, fieldbackground=FIELD, foreground=TEXT, rowheight=29)
        s.configure("Treeview.Heading", background=CARD, foreground=BLUE, font=("Segoe UI", 10, "bold"))
        s.map("Treeview", background=[("selected", "#244d70")])

    def card(self, parent, title):
        frame = ttk.Frame(parent, style="Card.TFrame", padding=10)
        ttk.Label(frame, text=title, background=CARD, foreground=BLUE,
                  font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0, 9))
        return frame

    def build(self):
        page = ttk.Frame(self.root, padding=20)
        page.pack(fill="both", expand=True)
        header = ttk.Frame(page)
        header.pack(fill="x", pady=(0, 8))
        ttk.Label(header, text="Text & Tone", font=("Segoe UI", 28, "bold")).pack(side="left")
        ttk.Label(header, text="  /  SIGNAL LAB", foreground=BLUE).pack(side="left", pady=(12, 0))
        ttk.Label(header, text="8 kHz  •  40 ms / character", style="Muted.TLabel").pack(side="right")
        top = ttk.Frame(page)
        top.pack(fill="x")
        top.columnconfigure(0, weight=3)
        top.columnconfigure(1, weight=2)
        inp = self.card(top, "01  /  SOURCE")
        inp.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        modes = ttk.Frame(inp, style="Card.TFrame")
        modes.pack(fill="x")
        for mode in ["Text", "WAV file"]:
            ttk.Radiobutton(modes, text=mode, variable=self.input_mode, value=mode,
                            command=self.change_source).pack(side="left", padx=(0, 16))
        line = ttk.Frame(inp, style="Card.TFrame")
        line.pack(fill="x", pady=(8, 5))
        ttk.Entry(line, textvariable=self.value).pack(side="left", fill="x", expand=True)
        self.browse_button = ttk.Button(line, text="Browse", command=self.browse, state="disabled")
        self.browse_button.pack(side="left", padx=(8, 0))
        ttk.Label(inp, text="English letters and spaces, or mono 8000 Hz encoded audio.",
                  style="Card.TLabel").pack(anchor="w")
        ttk.Label(inp, text="Expected text for WAV accuracy (optional; spaces are preserved)",
                  style="Card.TLabel").pack(anchor="w", pady=(5, 0))
        ttk.Entry(inp, textvariable=self.expected).pack(fill="x")
        dec = self.card(top, "02  /  DECODER")
        dec.grid(row=0, column=1, sticky="nsew")
        for mode, label in [("FFT", "Fourier transform"), ("Filter bank", "Resonator filter bank"),
                            ("Both", "Compare both methods")]:
            ttk.Radiobutton(dec, text=label, variable=self.decoder_mode, value=mode).pack(anchor="w")
        display = self.card(page, "DECODED MESSAGE  /  LCD")
        display.pack(fill="both", expand=True, pady=8)
        self.lcd = tk.Text(display, height=2, wrap="word", bg="#061b27", fg="#70dbff",
                           font=("Consolas", 16), relief="flat", padx=15, pady=12,
                           selectbackground="#245b78", insertbackground=BLUE)
        self.lcd.pack(fill="both", expand=True)
        self.show_lcd("READY\nYour recovered message will appear here.")
        ttk.Label(display, textvariable=self.metrics, style="Card.TLabel").pack(anchor="w", pady=(8, 0))
        filters = self.card(page, "03  /  FILTER INSPECTOR")
        filters.pack(fill="x")
        checkrow = ttk.Frame(filters, style="Card.TFrame")
        checkrow.pack(fill="x")
        for index, f in enumerate(FREQUENCIES):
            ttk.Checkbutton(checkrow, text=f"{f} Hz", variable=self.filters[f]).grid(
                row=index // 6, column=index % 6, sticky="w", padx=(0, 22))
        opts = ttk.Frame(filters, style="Card.TFrame")
        opts.pack(fill="x", pady=(8, 0))
        ttk.Button(opts, text="All", style="Secondary.TButton", command=lambda: self.select_filters(True)).pack(side="left")
        ttk.Button(opts, text="Clear", style="Secondary.TButton", command=lambda: self.select_filters(False)).pack(side="left", padx=6)
        ttk.Label(opts, text="Character", style="Card.TLabel").pack(side="left", padx=(12, 6))
        self.spinner = ttk.Spinbox(opts, from_=1, to=1, textvariable=self.segment_number, width=5)
        self.spinner.pack(side="left", padx=(0, 16))
        for name, var in self.filter_views.items():
            ttk.Checkbutton(opts, text=name, variable=var).pack(side="left", padx=7)
        ttk.Label(filters, text="Time and spectrum show filtered samples from the selected character.",
                  style="Card.TLabel").pack(anchor="w", pady=(8, 0))
        messages = self.card(page, "04  /  MESSAGE COMPARISON")
        messages.pack(fill="x", pady=8)
        checks = ttk.Frame(messages, style="Card.TFrame")
        checks.pack(fill="x")
        for name, var in self.message_views.items():
            ttk.Checkbutton(checks, text=name, variable=var).pack(side="left", padx=(0, 22))
        ttk.Label(messages, text="Recovered audio is re-encoded decoded text. Accuracy requires known expected text.",
                  style="Card.TLabel").pack(anchor="w", pady=(5, 0))
        actions = ttk.Frame(page)
        actions.pack(fill="x", pady=(0, 6))
        self.encode_button = ttk.Button(actions, text="Encode", command=self.encode)
        self.run_button = ttk.Button(actions, text="Decode", command=self.run)
        self.plot_button = ttk.Button(actions, text="Plot selected", command=self.plot)
        self.evaluate_button = ttk.Button(actions, text="Evaluation", command=self.evaluate)
        for b in [self.encode_button, self.run_button, self.plot_button, self.evaluate_button]:
            b.pack(side="left", padx=(0, 8))
        for label, command in [("Play", self.play), ("Stop", sd.stop)]:
            ttk.Button(actions, text=label, style="Secondary.TButton", command=command).pack(side="left", padx=(0, 8))
        ttk.Button(actions, text="Exit", style="Secondary.TButton", command=self.close).pack(side="right")
        ttk.Label(page, textvariable=self.status, style="Muted.TLabel", wraplength=1020).pack(anchor="w")

    def show_lcd(self, text):
        self.lcd.configure(state="normal")
        self.lcd.delete("1.0", "end")
        self.lcd.insert("1.0", text)
        self.lcd.configure(state="disabled")

    def select_filters(self, value):
        for var in self.filters.values():
            var.set(value)

    def change_source(self):
        self.value.set("")
        self.expected.set("")
        self.browse_button.configure(state="normal" if self.input_mode.get() == "WAV file" else "disabled")

    def browse(self):
        path = filedialog.askopenfilename(filetypes=[("WAV audio", "*.wav")])
        if path:
            self.value.set(path)
            self.expected.set("")

    def background(self, function, success, message):
        if self.future is not None:
            return
        for button in [self.encode_button, self.run_button, self.evaluate_button, self.plot_button]:
            button.configure(state="disabled")
        self.status.set(message)
        self.future = self.pool.submit(function)
        self.root.after(80, lambda: self.poll(success))

    def poll(self, success):
        if self.closed:
            return
        if not self.future.done():
            self.root.after(80, lambda: self.poll(success))
            return
        future, self.future = self.future, None
        for button in [self.encode_button, self.run_button, self.evaluate_button, self.plot_button]:
            button.configure(state="normal")
        try:
            success(future.result())
        except Exception as error:
            self.status.set(str(error))
            if self.result is None:
                self.show_lcd("CHECK INPUT\n" + str(error))
            messagebox.showerror("Unable to complete", str(error), parent=self.root)

    def encode(self):
        if self.future is not None:
            return
        if self.input_mode.get() != "Text":
            messagebox.showinfo("Text required", "Choose Text and enter a message to encode.")
            return
        text = self.value.get()
        from Core.Encoder import validator
        try:
            validator(text, self.engine.signals)
        except Exception as error:
            messagebox.showerror("Invalid text", str(error), parent=self.root)
            return
        path = filedialog.asksaveasfilename(parent=self.root, title="Save encoded WAV",
            defaultextension=".wav", initialfile="encoded.wav", filetypes=[("WAV audio", "*.wav")])
        if not path:
            return
        def saved(result):
            self.result = result
            self.input_mode.set("WAV file")
            self.value.set(path)
            self.expected.set(text)
            self.browse_button.configure(state="normal")
            self.spinner.configure(to=len(result.segments))
            self.segment_number.set(1)
            self.show_lcd("ENCODED AND SAVED\nPress Decode to recover your message.")
            self.metrics.set(f"{len(text)} characters • {len(result.audio):,} samples • {len(result.audio)/result.fs:.2f} seconds")
            self.status.set(f"Saved WAV: {path}")
        self.background(lambda: self.engine.encode_file(text, path), saved, "Encoding and saving WAV…")

    def run(self):
        if self.future is not None:
            return
        if self.input_mode.get() != "WAV file":
            messagebox.showinfo("Encode first", "Press Encode to save your text as WAV, or select a WAV file.")
            return
        value, decoder = self.value.get(), self.decoder_mode.get()
        expected = self.expected.get() or None
        self.result = None
        self.show_lcd("PROCESSING...")
        self.background(lambda: self.engine.decode_file(value, decoder, expected), self.decoded,
                        "Reading WAV, decoding, and calculating evaluation…")

    def decoded(self, output):
        result, records = output
        self.records.extend(records)
        self.completed(result)
        self.status.set(self.status.get() + "  |  Results saved. Press Evaluation to view or export.")

    def completed(self, result):
        self.result = result
        self.spinner.configure(to=len(result.segments))
        self.segment_number.set(1)
        self.show_lcd("\n".join(f"{name.upper()}  >  {text}" for name, text in result.decoded.items()))
        self.metrics.set(f"{len(result.segments)} characters  •  {len(result.audio):,} samples  •  "
                         f"{len(result.audio) / result.fs:.2f} seconds  •  {result.fs:,} Hz")
        details = []
        for name, text in result.decoded.items():
            check = ""
            if result.source_text is not None:
                check = " · exact match" if text == result.source_text else " · text mismatch"
            details.append(f"{name}: {result.times[name] * 1000:.1f} ms{check}")
        self.status.set("  |  ".join(details))

    def evaluate(self):
        if not self.records:
            messagebox.showinfo("No evaluations", "Decode a WAV first. Its results will appear here.")
            return
        self.show_evaluation(list(self.records))

    def show_evaluation(self, records):
        window = tk.Toplevel(self.root)
        window.title("Evaluation | Text & Tone")
        window.geometry("1020x720")
        window.minsize(980, 680)
        window.configure(bg=BG)
        page = ttk.Frame(window, padding=20)
        page.pack(fill="both", expand=True)
        ttk.Label(page, text="Decoder evaluation", font=("Segoe UI", 22, "bold")).pack(anchor="w")
        ttk.Label(page, text="Actual WAV decoding runs from this session, including repeated runs.\n"
                  "Accuracy uses expected text, including case and spaces. Unknown references show N/A.",
                  style="Muted.TLabel").pack(anchor="w", pady=(7, 15))
        summary = []
        for method in ["FFT", "Filter bank"]:
            group = [r for r in records if r["decoder"] == method and r["total"] is not None]
            correct, total = sum(r["correct"] for r in group), sum(r["total"] for r in group)
            summary.append(f"{method}: {100 * correct / total:.2f}% ({correct}/{total})" if total else f"{method}: N/A")
        ttk.Label(page, text="     |     ".join(summary), foreground=BLUE,
                  font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(0, 15))
        columns = ("test", "decoder", "accuracy", "characters", "time", "status")
        table_frame = ttk.Frame(page)
        table_frame.pack(fill="both", expand=True)
        table = ttk.Treeview(table_frame, columns=columns, show="headings", height=14)
        scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=table.yview)
        scrollbar.pack(side="right", fill="y")
        table.configure(yscrollcommand=scrollbar.set)
        for col, title, width in zip(columns, ["Test", "Decoder", "Accuracy", "Correct / total", "Time (ms)", "Result"],
                                     [245, 110, 90, 115, 100, 100]):
            table.heading(col, text=title)
            table.column(col, width=width, minwidth=70)
        table.pack(side="left", fill="both", expand=True)
        for r in records:
            table.insert("", "end", values=(r["test"], r["decoder"],
                f'{100*r["accuracy"]:.2f}%' if r["accuracy"] is not None else "N/A",
                f'{r["correct"]}/{r["total"]}' if r["total"] is not None else "N/A", f'{r["seconds"]*1000:.2f}',
                "UNSCORED" if r["expected"] is None else "PASS" if r["exact_match"] else "MISMATCH"))
        buttons = ttk.Frame(page)
        buttons.pack(fill="x", pady=(14, 0))
        ttk.Button(buttons, text="Export CSV", command=lambda: self.export_evaluation(records)).pack(side="left")
        ttk.Button(buttons, text="Close", style="Secondary.TButton", command=window.destroy).pack(side="right")
        self.status.set("Evaluation complete. " + " | ".join(summary))

    def export_evaluation(self, records=None):
        records = self.records if records is None else records
        if not records:
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", initialfile="decoder_evaluation.csv",
                                           filetypes=[("CSV", "*.csv")])
        if path:
            try:
                with open(path, "w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(records[0]))
                    writer.writeheader()
                    writer.writerows(records)
            except OSError as error:
                messagebox.showerror("Save failed", str(error))

    def plot(self):
        if self.result is None:
            messagebox.showinfo("No result", "Run a message or WAV first.")
            return
        filters = [f for f, var in self.filters.items() if var.get()]
        fv = [v for v, var in self.filter_views.items() if var.get()]
        mv = [v for v, var in self.message_views.items() if var.get()]
        try:
            if fv and not filters:
                raise ValueError("Select at least one filter frequency.")
            if not mv and not (filters and fv):
                raise ValueError("Choose a time, spectrum, or pole-zero plot.")
            make_plots(self.result, self.engine, filters, fv, mv, self.segment_number.get() - 1)
            plt.show(block=False)
        except Exception as error:
            messagebox.showerror("Cannot plot", str(error))

    def play(self):
        if self.result is None:
            messagebox.showinfo("No audio", "Run an input first.")
            return
        try:
            sd.play(self.result.audio, self.result.fs)
        except Exception as error:
            messagebox.showerror("Playback unavailable", str(error))

    def save(self):
        if self.result is None:
            messagebox.showinfo("No audio", "Run an input first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".wav", initialfile="encoded.wav",
                                           filetypes=[("WAV audio", "*.wav")])
        if path:
            try:
                wavfile.write(path, self.result.fs, self.result.audio.astype(np.float32))
                self.status.set(f"Saved to {path}")
            except Exception as error:
                messagebox.showerror("Save failed", str(error))

    def close(self):
        self.closed = True
        sd.stop()
        plt.close("all")
        self.pool.shutdown(wait=False, cancel_futures=True)
        self.root.destroy()


def main():
    configure_dpi()
    root = tk.Tk()
    try:
        DSPInterface(root)
    except Exception as error:
        root.withdraw()
        messagebox.showerror("Startup error", str(error))
        root.destroy()
        return
    root.mainloop()


if __name__ == "__main__":
    main()
