# Text & Tone Lab

Dark desktop interface with sky-blue controls for the DSP project.

## Start

Double-click **Start DSP Lab.bat**, or run `GUI.py` / `main.py` using the Python environment in `.venv`.
On another computer install `requirements.txt` into your environment first. Tkinter is also required (included with standard Windows Python).

## Use

1. Choose Text or WAV file. Text accepts English letters and spaces. WAV files must be mono, 8000 Hz, with complete 320-sample characters.
2. Choose FFT, Filter bank, or Both. Click Run decoder. Results appear in the blue LCD display.
3. Play the input or save it as a WAV using the bottom controls.
4. Select individual filters (or All), a character number, and Time / Spectrum / Pole-zero. Click Plot selected.
5. Select Input / Recovered time or spectrum for message plots. Recovered signals are re-encoded decoded text, not filtered reconstructions of the input audio.
6. Evaluate runs both decoders on seven labelled clean test messages, including every English letter in both cases and spaces. It reports character accuracy and computation time, and offers CSV export. It does not establish noise robustness or accuracy on instructor-provided recordings.
7. Exit stops playback and closes the app and its plots.

## Files

- `GUI.py`: interface and controls; work runs off the UI thread.
- `app_engine.py`: input validation, decoding coordination, evaluation.
- `plots.py`: time, one-sided amplitude spectrum, and pole-zero charts.
- `data/`: copies of your MAT signal bank and frequency table.
- Existing encoder/decoder modules: staged copies of your work.

The original DSP project was not edited. In these copies, the Resonator coefficient assignment is corrected to `b, a = iirpeak(...)`. The data loader now resolves files relative to this folder. `main.py` starts the GUI.

Filter time and spectrum views show the selected character after each filter. Spectrum amplitudes preserve DC/Nyquist scaling. Ten second-order resonators use 25 Hz bandwidth; the 4000 Hz detector uses a second-order highpass with a 3900 Hz cutoff.

Tests: `.venv\Scripts\python.exe -m unittest test_app -v`
