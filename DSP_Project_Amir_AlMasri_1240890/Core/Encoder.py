import numpy as np
import scipy.io.wavfile
import sounddevice
from Core.Data_Loader import loading_signals


class EmptyString(ValueError):
    pass
class InvalidCharacterError(ValueError):
    pass


def validator(input_string: str, signals: dict) -> None:
    """Check that the input is nonempty and contains supported characters.

    Raises EmptyString for empty input and InvalidCharacterError
    for the first character missing from the signal dictionary.
    """
    if input_string =="": raise EmptyString("The input can't be empty.")
    for c in input_string:
        if c not in signals:
            raise InvalidCharacterError(
                f"Unsupported character: {c!r}. "
                "Only English letters and spaces are allowed."
            )

def encoder(validated_input:str,signals:dict)->np.ndarray:
    """Convert validated text into one audio signal.

    Retrieves each character's samples from the signal dictionary
    and concatenates them in input order.
    Returns a one-dimensional NumPy array.
    """
    segments = [signals[c] for c in validated_input]
    text_signal = np.concatenate(segments)
    return text_signal

def audio_saver(filename:str,text_signal:np.ndarray,Fs:int)->None:
    """Save the encoded signal as a 32-bit floating-point WAV file.

    Uses Fs as the sampling frequency in Hz.
    """
    scipy.io.wavfile.write(filename,Fs,text_signal.astype(np.float32))

def sound_player(text_signal:np.ndarray,Fs:int)->None:
    """Play the signal at Fs samples per second and wait until it finishes."""
    sounddevice.play(text_signal,Fs)
    sounddevice.wait()

def encoder_runner(output_filename: str) -> bool:
    """Run the text-to-audio encoding process.

    Loads character signals, reads and validates user input,
    concatenates the corresponding samples, and saves a WAV file.
    Verifies the saved audio and plays it.
    Returns False if loading or validation fails, otherwise True.
    """

    generator = loading_signals()
    if generator is None:
        return False

    signals, Fs, Tc = generator
    input_str = input("Please enter a string: ")

    try:
        validator(input_str, signals)
    except (EmptyString, InvalidCharacterError) as e:
        print(e)
        return False

    text_signal = encoder(input_str, signals)
    expected_samples = len(input_str) * round(Fs * Tc)

    assert len(text_signal) == expected_samples

    audio_saver(output_filename, text_signal, Fs)
    saved_Fs, saved_signal = scipy.io.wavfile.read(output_filename)

    assert saved_Fs == Fs
    assert np.array_equal(
        saved_signal, text_signal.astype(np.float32)
    )

    print("Input:", repr(input_str))
    print("Signal shape:", text_signal.shape)
    print("Expected samples:", expected_samples)
    print("Duration:", len(text_signal) / Fs, "seconds")
    print("Encoding Done")

    sound_player(saved_signal, saved_Fs)
    return True
