import string
from pathlib import Path
from scipy.io import loadmat
import numpy as np


class EmptyFileError(ValueError):
    pass

def loading_signals()-> tuple[dict[str, np.ndarray], int, float] | None:
    """Load character waveforms and sampling settings from the MAT file.

    Maps the signal columns to lowercase letters, uppercase letters,
    and space. Returns the character-to-samples dictionary, sampling
    frequency Fs, and character duration Tc.
    Prints an error and returns None if the file is missing.
    """
    try:
        data = loadmat(Path(__file__).resolve().parents[1] / "Matlab Work" / "character_signals.mat")
    except FileNotFoundError:
        print("File {character_signals.mat} Not found.")
    else:
        signals = data["signals"]
        Fs = int(data["Fs"].item())
        Tc = float(data["Tc"].item())
        characters = string.ascii_lowercase + string.ascii_uppercase + " "
        signal_dict = {}

        for index, character in enumerate(characters):
            signal_dict[character] = signals[:, index]

        return signal_dict, Fs, Tc


def loading_frequencies() -> dict[str, list[int]] | None:
    """Load the character frequency table from a colon-separated text file.

    Expects 27 rows ordered a-z, then space, with three frequencies
    per row. Returns a dictionary mapping each character to its
    [low, middle, high] frequencies.
    Prints an error and returns None if the file is missing,
    empty, incorrectly sized, or contains invalid numeric values.
    """

    characters = string.ascii_lowercase + " "

    try:
        with open(Path(__file__).resolve().parents[1] / "Matlab Work" / "frequencies.txt", "r") as f:
            content = f.read()

        if not content.strip():
            raise ValueError("The frequency file is empty.")

        lines = content.strip().splitlines()

        if len(lines) != 27:
            raise ValueError("Expected 27 rows: a–z, then space.")

        frequency_lookup_table = {}

        for character, line in zip(characters, lines):
            freq_row = [int(value) for value in line.split(":")]

            if len(freq_row) != 3:
                raise ValueError(
                    f"Expected 3 frequencies for {character!r}."
                )

            frequency_lookup_table[character] = freq_row

        return frequency_lookup_table

    except FileNotFoundError:
        print("File frequencies.txt not found.")
    except ValueError as error:
        print(f"Invalid frequency file: {error}")

    return None
