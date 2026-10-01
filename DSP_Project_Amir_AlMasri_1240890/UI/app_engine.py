"""Validated input handling and repeatable evaluation for the desktop interface."""
from dataclasses import dataclass
from time import perf_counter
import string
import hashlib
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from Core.Data_Loader import loading_signals, loading_frequencies
from Core.Encoder import encoder, validator
from Core.FFT_Decoder import FFT_decoder_runner
from Core.Filter_Decoder import filter_decoder_runner


@dataclass
class Result:
    audio: np.ndarray
    segments: np.ndarray
    fs: int
    source_text: str | None
    decoded: dict
    times: dict


class Engine:
    def __init__(self):
        loaded = loading_signals()
        self.lookup = loading_frequencies()
        if loaded is None or self.lookup is None:
            raise ValueError("The character signal bank or frequency table could not be loaded.")
        self.signals, self.fs, self.tc = loaded
        self.saved_texts = {}
        if self.fs != 8000 or not np.isclose(self.tc, .04):
            raise ValueError("The signal bank must use 8000 Hz and 40 ms characters.")
        if len(self.signals) != 53 or any(v.shape != (320,) for v in self.signals.values()):
            raise ValueError("Expected 53 character signals with 320 samples each.")
        if any(not np.all(np.isfinite(v)) for v in self.signals.values()):
            raise ValueError("Invalid samples in the character bank.")

    @staticmethod
    def read_wav(filename):
        fs, raw = wavfile.read(filename)
        if raw.ndim != 1:
            raise ValueError("Select a mono WAV, not a stereo recording.")
        if raw.dtype == np.uint8:
            audio = (raw.astype(float) - 128) / 128
        elif np.issubdtype(raw.dtype, np.signedinteger):
            info = np.iinfo(raw.dtype)
            audio = raw.astype(float) / max(abs(int(info.min)), int(info.max))
        else:
            audio = raw.astype(float)
        return fs, audio

    @staticmethod
    def split(audio, fs, tc):
        N = round(fs * tc)
        if not len(audio) or not np.all(np.isfinite(audio)):
            raise ValueError("Audio is empty or contains invalid sample values.")
        if len(audio) % N:
            raise ValueError(f"The WAV ends with an incomplete {N}-sample character.")
        segments = audio.reshape(-1, N)
        if np.any(np.max(np.abs(segments), axis=1) == 0):
            raise ValueError("A silent segment cannot identify a character. Encoded spaces contain tones.")
        return segments

    def decode(self, audio, mode, fs, source_text=None):
        segments = self.split(audio, fs, self.tc)
        decoded, times = {}, {}
        methods = {"FFT": FFT_decoder_runner, "Filter bank": filter_decoder_runner}
        for name, function in methods.items():
            if mode not in (name, "Both"):
                continue
            start = perf_counter()
            decoded[name] = function(self.lookup, segments, fs)  # pass real fs
            times[name] = perf_counter() - start
        return Result(audio, segments, fs, source_text, decoded, times)

    def process(self, input_mode, value, decoder):
        if input_mode == "Text":
            validator(value, self.signals)
            audio = encoder(value, self.signals).astype(np.float32)
            return self.decode(audio, decoder, self.fs, value)
        fs, audio = self.read_wav(value.strip().strip('"'))
        return self.decode(audio, decoder, fs)

    def reconstruct(self, text):
        return encoder(text, self.signals)

    def encode_file(self, text, filename):
        """Save a WAV and remember its reference text for this session."""
        validator(text, self.signals)
        audio = self.reconstruct(text).astype(np.float32)
        wavfile.write(filename, self.fs, audio)
        self.saved_texts[hashlib.sha256(Path(filename).read_bytes()).hexdigest()] = text
        return Result(audio, self.split(audio, self.fs, self.tc), self.fs, text, {}, {})

    def decode_file(self, filename, mode, expected=None):
        """Read the actual WAV and evaluate only against known reference text."""
        filename = str(filename).strip().strip('"')
        if expected is None:
            expected = self.saved_texts.get(hashlib.sha256(Path(filename).read_bytes()).hexdigest())
        if expected is not None:
            validator(expected, self.signals)
        fs, audio = self.read_wav(filename)
        result = self.decode(audio, mode, fs, expected)
        records = []
        for name, recovered in result.decoded.items():
            correct = sum(a == b for a, b in zip(expected, recovered)) if expected is not None else None
            total = len(expected) if expected is not None else None
            records.append(dict(test=Path(filename).name, filename=str(Path(filename).resolve()),
                decoder=name, expected=expected, recovered=recovered, correct=correct, total=total,
                accuracy=correct / total if total else None,
                exact_match=recovered == expected if expected is not None else None,
                decoded_characters=len(recovered), samples=len(audio), sampling_rate=fs,
                duration=len(audio)/fs, seconds=result.times[name], error=""))
        return result, records

    def evaluate(self):
        """Measure both methods on labeled, clean synthetic messages only."""
        messages = [
            ("Lowercase alphabet", string.ascii_lowercase),
            ("Uppercase alphabet", string.ascii_uppercase),
            ("Mixed case", "Hi a Hello World"),
            ("Repeated letters and spaces", "aaAA  zzZZ   Hello"),
            ("Boundary spaces", "  a A  "),
            ("Space", " "),
            ("Long message", ("Digital Signal Processing " * 20)),
        ]
        records = []
        for label, text in messages:
            audio = self.reconstruct(text).astype(np.float32)
            segments = self.split(audio, self.fs, self.tc)
            for name, function in [("FFT", FFT_decoder_runner), ("Filter bank", filter_decoder_runner)]:
                start = perf_counter()
                try:
                    recovered = function(self.lookup, segments, self.fs)
                    elapsed = perf_counter() - start
                    correct = sum(a == b for a, b in zip(text, recovered))
                    error = ""
                except Exception as exc:
                    elapsed, recovered, correct, error = perf_counter() - start, "", 0, str(exc)
                records.append(dict(test=label, decoder=name, expected=text, recovered=recovered,
                                    correct=correct, total=len(text), accuracy=correct / len(text),
                                    seconds=elapsed, error=error))
        return records