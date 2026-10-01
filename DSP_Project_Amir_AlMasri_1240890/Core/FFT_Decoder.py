import numpy as np
import scipy

def FFT_decoder_runner(frequency_lookup_table:dict,segments:np.ndarray,Fs:int) -> str | None:
    """Decode audio segments into text using Fourier analysis.

    Detects each segment's case, low, middle, and high frequencies
    from FFT magnitudes, then maps them to characters using the
    frequency lookup table. Returns the recovered message.
    """
    rows = frequency_detector(segments, Fs)
    return message_extractor(rows, frequency_lookup_table)


def load_segments(filename:str,Tc:float):
    """Read a mono WAV file and divide it into character segments.

    Calculates segment length from the sampling rate and Tc.
    Rejects an incomplete final segment. Returns the sampling
    frequency and a matrix containing one segment per row.
    """
    Fs, audio = scipy.io.wavfile.read(filename)
    N = round(Fs*Tc)

    if len(audio)%N !=0:
        raise ValueError("Audio contains an incomplete character segments.")

    segments = audio.reshape(-1,N)

    return Fs,segments

def message_extractor(rows: np.ndarray, frequency_lookup_table: dict) -> str:
    """Convert detected frequency rows into a text message.

    Matches each low, middle, and high frequency combination
    to a character, then applies the detected case.
    Raises ValueError when a combination is not in the table.
    Returns the joined characters.
    """
    rows_freq = rows[:, 1:]
    cases = rows[:, 0]
    extracted_char = []

    for row, case in zip(rows_freq, cases):
        for c, frequencies in frequency_lookup_table.items():
            if np.array_equal(row, frequencies):
                if case == 200:
                    c = c.upper()

                extracted_char.append(c)
                break
        else:
            raise ValueError(f"Unknown frequencies: {row}")

    return "".join(extracted_char)

def frequency_detector(segments: np.ndarray, Fs: int) -> np.ndarray:
    """Detect the encoding frequencies of each character segment.

    Computes FFT magnitudes and selects the strongest candidate
    in each frequency group. Returns one row per segment in the
    order [fCase, fLow, fMid, fHigh].
    """
    rows = []
    for segment in segments:
        mag, frequencies = frequency_components(segment, Fs)

        fCase = 200 if uppercase_detector(mag) else 100
        fLow = low_frequency_detector(mag)
        fMid = mid_frequency_detector(mag)
        fHigh = high_frequency_detector(mag)

        rows.append([fCase, fLow, fMid, fHigh])

    return np.array(rows)

def frequency_components(segment: np.ndarray, Fs: int):
    """Calculate the frequency content of one character segment.

    Returns two arrays: raw FFT magnitudes and their corresponding
    frequencies in Hz, using the real-input FFT.
    """
    spectrum = np.fft.rfft(segment)
    mag = np.abs(spectrum)
    frequencies = np.fft.rfftfreq(len(segment),d=1/Fs)
    return mag,frequencies

def uppercase_detector(mag: np.ndarray) -> bool:
    """Determine character case from the 100 Hz and 200 Hz magnitudes.

    Returns False when 100 Hz is stronger, otherwise True.
    Assumes 320-sample segments at an 8000 Hz sampling rate.
    """
    if mag[4] > mag[8]:
        return False
    return True

def low_frequency_detector(mag: np.ndarray) -> int:
    """Return the strongest low-group frequency: 400, 600, or 1000 Hz.

    Assumes 320-sample segments at an 8000 Hz sampling rate.
    """
    values = [mag[16], mag[24], mag[40]]
    max_power = max(values)
    idx = values.index(max_power)
    return [400,600,1000][idx]

def mid_frequency_detector(mag: np.ndarray) -> int:
    """Return the strongest middle-group frequency: 800, 1200, or 2000 Hz.

    Assumes 320-sample segments at an 8000 Hz sampling rate.
    """
    values = [mag[32], mag[48], mag[80]]
    max_power = max(values)
    idx = values.index(max_power)
    return [800,1200,2000][idx]

def high_frequency_detector(mag: np.ndarray) -> int:
    """Return the strongest high-group frequency: 1600, 2400, or 4000 Hz.

    Assumes 320-sample segments at an 8000 Hz sampling rate.
    """
    values = [mag[64], mag[96], mag[160]]
    max_power = max(values)
    idx = values.index(max_power)
    return [1600,2400,4000][idx]