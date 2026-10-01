# DSP_Text_Processing

### Text-to-Audio Encoding and Audio-to-Text Decoding

**Developed by Amir AlMasri — Birzeit University**  
**Technologies: MATLAB, Simulink, Python, NumPy, SciPy, Matplotlib, and Tkinter**

## Project overview

For my Digital Signal Processing course project, I developed a system that converts English text into a sequence of audio tones and reconstructs the original text from an encoded WAV file.

The system supports 53 characters: lowercase letters, uppercase letters, and spaces. It represents each character using a specific combination of frequencies. This is a tone-based communication system: the audio carries encoded characters rather than spoken words.

I built the signal-generation stage in MATLAB and Simulink, then implemented message encoding and two different decoding methods in Python. The project connects signal synthesis, frequency analysis, digital filtering, file handling, and performance evaluation in one application.

## Signal generation in MATLAB and Simulink

I began by designing a Simulink model containing four sinusoidal sources: one for letter case and three for character identification. I configured the sources to generate cosine signals and added their outputs to produce one character waveform.

The frequency groups are:

| Component | Candidate frequencies |
|---|---|
| Letter case | 100 Hz for lowercase; 200 Hz for uppercase |
| Low-frequency group | 400, 600, 1000 Hz |
| Middle-frequency group | 800, 1200, 2000 Hz |
| High-frequency group | 1600, 2400, 4000 Hz |

The three character frequencies follow the supplied lookup table. For example, lowercase **a** uses 100, 400, 800, and 1600 Hz. Uppercase **A** uses the same character frequencies with a 200 Hz case tone.

Each cosine has an amplitude of 0.25. Their sum therefore remains within an absolute amplitude of one. The sampling frequency is 8000 Hz, and each character lasts 40 milliseconds:

N = Fs × Tc
N = 8000 × 0.04 = 320 samples

I wrote a MATLAB script that reads the frequency table, assigns the appropriate frequencies, runs the Simulink model for each character, and collects its output. The script saves the resulting signals in a MAT file containing a **320 × 53 matrix**, together with character labels, sampling frequency, and character duration.

This reusable signal bank allows Python to encode new messages without running Simulink again.

## Text encoding in Python

I implemented data-loading functions to import the MATLAB signals and frequency table. The character waveforms are stored in a Python dictionary, where each character maps to its NumPy sample array.

The encoder validates the input and rejects empty text or unsupported characters, such as digits and punctuation. It retrieves the waveform for each character and concatenates the arrays in message order.

For example, **“Hi a”** contains four characters, including the space, so its encoded signal contains 1280 samples and lasts 0.16 seconds.

The complete waveform is saved as a WAV file and can be played through the application. Spaces have their own frequency combination; they are encoded signals rather than silent gaps.

## FFT-based decoding

I implemented a decoder that reads an encoded WAV file and divides the waveform into character segments. For the project’s 8000 Hz sampling rate, each segment contains 320 samples.

For each segment, the decoder computes a real-input FFT and examines its magnitude spectrum. The frequency-bin spacing is:

Δf = Fs / N
Δf = 8000 / 320 = 25 Hz

All prescribed frequencies align with this grid. The decoder selects the strongest candidate separately within the case, low, middle, and high groups.

The detected character-frequency triplet is matched against the lookup table, and the case tone determines whether the letter is uppercase or lowercase. The recovered characters are joined to reconstruct the message while preserving spaces.

## Filter-bank decoding

I also implemented a second decoder based on digital filtering.

I designed a Resonator class using second-order IIR peak filters. Each resonator targets one candidate frequency below 4000 Hz, with a bandwidth of 25 Hz. Its quality factor is calculated as:

Q = f0 / BW

The 4000 Hz component lies at the Nyquist frequency. I handled this branch using a second-order Butterworth highpass filter with a 3900 Hz cutoff.

Each character segment passes through the filter bank. The decoder calculates the energy of every branch by summing the squared output samples:

E = y[0]² + y[1]² + ... + y[N−1]²

It selects the highest-energy candidate within each frequency group and uses the same character lookup procedure as the FFT decoder. The filter-bank decisions use filtered-signal energy, without an FFT.

## Application interface and visualization

The desktop interface brings the project’s functions together using a dark theme, sky-blue controls, and an LCD-style decoded-text display.

Its main features include:

- **Encode:** validates text, asks for a filename, and saves the encoded WAV.
- **Decode:** loads a WAV and runs the FFT decoder, filter-bank decoder, or both.
- **Evaluation:** displays decoding results and exports them to CSV.
- **Playback:** plays and stops the encoded audio.
- **Filter inspection:** displays selected filter outputs in the time and frequency domains, together with pole-zero diagrams.
- **Message comparison:** displays the input waveform and a waveform reconstructed by re-encoding the decoded text.

Accuracy is calculated when the original text is known. For external WAV files without reference text, the application reports accuracy as unavailable rather than assuming the decoded message is correct.

## Testing and measured results

I tested the complete workflow by encoding messages, saving them as WAV files, loading those files again, and decoding them using both methods.

The final evaluation included eight messages:

| Test | Characters |
|---|---:|
| Lowercase alphabet | 26 |
| Uppercase alphabet | 26 |
| Mixed-case message | 16 |
| Repeated letters and multiple spaces | 18 |
| Leading and trailing spaces | 7 |
| Single space | 1 |
| Repeating letter pattern | 101 |
| Long personal message | 576 |
| **Total per decoder** | **771** |

Both decoders recovered **771 out of 771 characters correctly**, including letter case and spaces. Each method recovered all eight messages exactly, producing 16 successful message-decoder runs.

These measurements apply to clean, generated WAV files with aligned character boundaries. They do not establish performance under background noise, microphone recording, timing errors, or frequency drift. The instructor-provided competition files were not included in these accuracy results.

## My contribution

I developed the project’s DSP components step by step:

- Designed and configured the Simulink character generator.
- Wrote the MATLAB automation script and generated the complete character-signal bank.
- Implemented Python data loading, input validation, message encoding, WAV saving, and playback.
- Implemented the FFT decoder and frequency-group detectors.
- Designed the resonator bank and implemented energy-based decoding.
- Implemented frequency-to-character matching and message reconstruction.
- Organized and refactored the application into separate core and interface modules.
- Ran the WAV-based evaluation and exported the measured results.

I used AI assistance to build the graphical interface, identify useful libraries, troubleshoot implementation issues, and prepare supporting documentation and presentation materials.

## What I learned

This project helped me connect DSP theory with a working application. I developed a clearer understanding of the relationship between time samples and FFT bins, the effect of sampling frequency, and the special treatment required at the Nyquist frequency.

Working with the filter bank taught me how bandwidth and quality factor affect selectivity and settling time. I also learned that irregular filter-output traces can result from startup transients and residual frequency components, rather than random noise.

Most importantly, I learned to verify the complete communication process: generate a signal, save it, load it independently, decode it, and compare the recovered message against the original.
