import numpy as np
from scipy.signal import sosfilt

from Core.FFT_Decoder import message_extractor
from Core.ResonatorBank import resonator_bank, Resonator


def filter_decoder_runner(frequency_lookup_table:dict,segments:np.ndarray,Fs:int)->str|None:
    """Decode audio segments into text using a filter bank.

    Builds the filters, detects each segment's frequency combination,
    and converts the detected frequencies into characters.
    """
    bank = resonator_bank(Fs=Fs,BW=25)
    rows = detect_filter_frequencies(segments,bank)
    return message_extractor(rows, frequency_lookup_table)


def filter_energies(segment: np.ndarray,bank:dict)->dict:
    """Calculate each filter's output energy for one character segment.

    Uses resonators for frequencies below 4000 Hz and an SOS highpass
    filter for 4000 Hz. Returns a dictionary mapping frequency to energy.
    """
    segment_energy_byf ={}
    for freq,fil in bank.items():
        if isinstance(fil,Resonator):
            segment_energy_byf[freq] = fil.energy(segment)

        else:
            out = sosfilt(fil, segment)
            segment_energy_byf[freq] = float(np.sum(out** 2))

    return segment_energy_byf


def detect_filter_frequencies(segments: np.ndarray, bank: dict) -> np.ndarray:
    """Identify the strongest frequency in each group for every segment.

    Compares filter output energies within the case, low, middle,
    and high groups. Returns one row per character in the order
    [fCase, fLow, fMid, fHigh].
    """
    rows = []
    for segment in segments:
        energies = filter_energies(segment, bank)

        fCase = max([100, 200], key=energies.get)
        fLow = max([400, 600, 1000], key=energies.get)
        fMid = max([800, 1200, 2000], key=energies.get)
        fHigh = max([1600, 2400, 4000], key=energies.get)

        rows.append([fCase, fLow, fMid, fHigh])

    return np.array(rows)
