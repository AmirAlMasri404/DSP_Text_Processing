import scipy.signal
import numpy as np
from scipy.signal import butter


class Resonator:

    def __init__(self,f0:int,BW:int,Fs:int):
        """Design a second-order resonator centered at f0.

        Calculates Q from f0 / BW and stores the filter coefficients.
        Frequencies and bandwidth are specified in Hz.
        """
        self.f0 = f0
        self.BW = BW
        self.Fs = Fs
        self.Q = f0/BW
        self.b, self.a = scipy.signal.iirpeak(f0, self.Q, fs=Fs)

    def apply(self, segment: np.ndarray) -> np.ndarray:
        """Filter one segment using the stored resonator coefficients.

        Starts from zero filter state and returns the filtered samples.
        """
        return scipy.signal.lfilter(self.b, self.a, segment)

    def energy(self, segment: np.ndarray) -> float:
        """Return the sum of squared samples after resonator filtering."""
        output = self.apply(segment)
        return float(np.sum(np.abs(output) ** 2))


def resonator_bank(Fs: int,BW: int = 25):
    """Create the frequency-selective filter bank.

    Returns a dictionary mapping ten target frequencies to
    Resonator objects and 4000 Hz to highpass SOS coefficients.
    BW specifies the resonators' bandwidth in Hz.
    """
    return {
        100: Resonator(100,BW,Fs),
        200: Resonator(200, BW, Fs),
        400:  Resonator(400, BW, Fs),
        600:  Resonator(600, BW, Fs),
        800:  Resonator(800, BW, Fs),
        1000: Resonator(1000, BW, Fs),
        1200: Resonator(1200, BW, Fs),
        1600: Resonator(1600, BW, Fs),
        2000: Resonator(2000, BW, Fs),
        2400: Resonator(2400, BW, Fs),
        4000: (
            Resonator(4000, BW, Fs)
            if Fs > 8000
            else butter(N=2, Wn=3900, btype="highpass",fs=Fs, output="sos")
            )
    }
