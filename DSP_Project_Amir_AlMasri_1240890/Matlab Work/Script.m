clear;
clc;

Fs = 8000;
Tc = 40e-3;
N = round(Fs * Tc);

folder = 'C:\Users\amirm\OneDrive\Desktop\DSP';

% File rows: a-z, then space. Columns: low, middle, high.
frequencies = readmatrix(fullfile(folder, 'frequencies.txt'),'Delimiter',':');

assert(isequal(size(frequencies), [27, 3]), ...
    'Frequency file must contain 27 rows and 3 columns.');
assert(all(isfinite(frequencies(:))), ...
    'Frequency file must contain only valid numbers.');

% Each column of signals corresponds to one character.
characters = ['a':'z', 'A':'Z', ' '];
signals = zeros(N, length(characters));

for k = 1:length(characters)

    character = characters(k);

    if character == ' '
        row = 27;
        fCase = 100;  % Space ignores case; use 100 Hz consistently.
    else
        row = double(lower(character)) - double('a') + 1;

        if character >= 'A' && character <= 'Z'
            fCase = 200;
        else
            fCase = 100;
        end
    end

    fLow  = frequencies(row, 1);
    fMid  = frequencies(row, 2);
    fHigh = frequencies(row, 3);

    SimOut = sim('character_encoder','ReturnWorkspaceOutputs', 'on');

    segment = SimOut.segment;

    assert(isnumeric(segment) && isvector(segment) ...
           && numel(segment) == N, ...
           'Expected 320 samples. Check Simulink output settings.');

    signals(:, k) = segment(:);

    fprintf('Generated character %d of %d\n', ...
            k, length(characters));
end

% Save in a format Python scipy.io.loadmat can read.
outputFile = fullfile(folder, 'character_signals.mat');

save(outputFile, 'signals', 'characters', 'Fs', 'Tc', '-v7');

fprintf('Saved signals to:\n%s\n', outputFile);
disp(size(signals));  % Expected: 320 × 53