"""Dark signal and pole-zero figures. Call from the GUI's main thread."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import sosfilt, sos2zpk, tf2zpk
from Core.ResonatorBank import resonator_bank

FREQUENCIES = [100, 200, 400, 600, 800, 1000, 1200, 1600, 2000, 2400, 4000]


def style():
    plt.rcParams.update({"figure.facecolor": "#101824", "axes.facecolor": "#152233",
                         "axes.edgecolor": "#415269", "axes.labelcolor": "#b8c9de",
                         "text.color": "#e5f0ff", "xtick.color": "#9aafc8",
                         "ytick.color": "#9aafc8", "grid.color": "#33455d",
                         "savefig.facecolor": "#101824", "font.size": 9})


def time_plot(ax, samples, fs, title):
    ax.plot(np.arange(len(samples)) / fs, samples, color="#55c6ff", linewidth=.8)
    ax.set(title=title, xlabel="Time (s)", ylabel="Amplitude")
    ax.grid(alpha=.4)


def spectrum_plot(ax, samples, fs, title):
    amplitude = np.abs(np.fft.rfft(samples)) / len(samples)
    amplitude[1:-1 if len(samples) % 2 == 0 else None] *= 2
    ax.plot(np.fft.rfftfreq(len(samples), 1 / fs), amplitude, color="#75e2c1", linewidth=.8)
    ax.set(title=title, xlabel="Frequency (Hz)", ylabel="One-sided amplitude")
    ax.grid(alpha=.4)


def pole_plot(ax, zeros, poles, title):
    theta = np.linspace(0, 2 * np.pi, 500)
    ax.plot(np.cos(theta), np.sin(theta), "--", color="#8395ab", linewidth=.8)
    ax.scatter(zeros.real, zeros.imag, facecolors="none", edgecolors="#55c6ff", s=65, label="Zeros")
    ax.scatter(poles.real, poles.imag, color="#ffbd7a", marker="x", s=60, label="Poles")
    ax.axhline(0, color="#415269", linewidth=.5)
    ax.axvline(0, color="#415269", linewidth=.5)
    ax.set(title=title, xlabel="Real", ylabel="Imaginary", xlim=(-1.15, 1.15), ylim=(-1.15, 1.15))
    ax.set_aspect("equal")
    ax.legend(facecolor="#152233", fontsize=8)
    ax.grid(alpha=.4)


def make_plots(result, engine, selected_filters, filter_views, message_views, segment_index):
    style()
    figures = []
    if selected_filters and filter_views:
        if not 0 <= segment_index < len(result.segments):
            raise ValueError("Choose an existing character segment.")
        bank = resonator_bank(result.fs, 25)
        segment = result.segments[segment_index]
        for start in range(0, len(selected_filters), 3):
            group = selected_filters[start:start + 3]
            fig, axes = plt.subplots(len(group), len(filter_views), squeeze=False,
                                     figsize=(4.6 * len(filter_views), 2.9 * len(group)), layout="constrained")
            fig.suptitle(f"Filter outputs • Character {segment_index + 1} • 25 Hz resonator bandwidth")
            figures.append(fig)
            for row, frequency in enumerate(group):
                item = bank[frequency]
                if isinstance(item, np.ndarray):  # SOS coefficients — the 8000 Hz highpass fallback
                    output = sosfilt(item, segment)
                    zeros, poles, _ = sos2zpk(item)
                else:  # a Resonator object — normal case, including 4000 Hz when Fs > 8000
                    output = item.apply(segment)
                    zeros, poles, _ = tf2zpk(item.b, item.a)
                for col, view in enumerate(filter_views):
                    ax, title = axes[row, col], f"{frequency} Hz • {view}"
                    if view == "Time":
                        time_plot(ax, output, result.fs, title)
                    elif view == "Spectrum":
                        spectrum_plot(ax, output, result.fs, title)
                    else:
                        pole_plot(ax, zeros, poles, title)
    jobs = []
    for view in ("Time", "Spectrum"):
        if f"Input {view.lower()}" in message_views:
            jobs.append(("Original input", result.audio, view))
        if f"Recovered {view.lower()}" in message_views:
            for name, text in result.decoded.items():
                jobs.append((f"Re-encoded decoded text • {name}", engine.reconstruct(text), view))
    if jobs:
        fig, axes = plt.subplots(len(jobs), 1, squeeze=False, figsize=(11, 2.5 * len(jobs)), layout="constrained")
        fig.suptitle("Original and reconstructed message")
        figures.append(fig)
        for ax, (label, samples, view) in zip(axes.flat, jobs):
            (time_plot if view == "Time" else spectrum_plot)(ax, samples, result.fs, label)
    return figures
