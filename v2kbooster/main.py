# coding: utf-8

import audiofile
import librosa
import numpy as np
import os
import pyloudnorm

from glob import glob
from scipy.signal import hilbert, savgol_filter
from sklearn.preprocessing import minmax_scale
from soundfile import LibsndfileError


def load_file(filename: str, mono: bool = False) -> (np.ndarray, int):
    try:
        signal, fs = audiofile.read(filename)
        if mono and len(signal.shape) == 2:
            signal = signal[0]
    except LibsndfileError:
        signal, fs = librosa.load(filename, mono=mono)

    return signal, fs


def interpolate(data: np.ndarray, to_size: int) -> np.ndarray:
    return np.interp(np.linspace(0, data.size - 1, num=to_size), np.arange(data.size), data)


def hilbert_frequency_rms(data: np.ndarray, fs: int, window_size: int = 16) -> np.ndarray:
    w = min(64, max(16, window_size // 32))
    f = (np.diff(np.unwrap(np.angle(hilbert(data)))) / (2.0 * np.pi) * fs)

    mf = []
    for x in range(0, f.size - w, w):
        mf.append(np.sqrt(np.mean(f[x:x + w]) ** 2))

    return 1 - savgol_filter(np.array(mf), w, 1)


def process_file(filename: str, weights: list, nfft: int, hw: float) -> (str, str):
    if '-enhanced' in filename:
        return None

    path, ext = os.path.splitext(filename)
    dir_name = os.path.dirname(path)
    new_filename = os.path.join(dir_name, f'{os.path.basename(path)}-enhanced{ext}')

    if os.path.isfile(new_filename):
        os.remove(new_filename)

    signal, fs = load_file(filename, mono=True)
    enhanced = minmax_scale(enhance_salient(signal, fs, weights, nfft, hw), (-1., 1.))

    audiofile.write(file=new_filename, signal=enhanced, sampling_rate=fs)

    return filename, new_filename


def enhance_salient(data: np.ndarray,
                    fs: int,
                    weights: list,
                    nfft: int,
                    hw: float) -> np.ndarray:
    harmonics = list(np.arange(1, len(weights)+1))

    harmonic_weights = ', '.join(list(map(lambda x: f'{x[0]}={x[1]}', zip(harmonics, weights))))

    print(f'harmonic weights: {harmonic_weights}')
    print(f'fft bins: {nfft}')

    spectrum = np.abs(librosa.stft(data, n_fft=nfft))

    spectrum = spectrum * (.5 * librosa.salience(spectrum,
                                                 freqs=librosa.fft_frequencies(sr=fs, n_fft=nfft),
                                                 harmonics=harmonics,
                                                 weights=weights,
                                                 fill_value=0))
    spectrum[0].fill(0.)

    signal = librosa.istft(spectrum, n_fft=nfft)

    envelope = interpolate((hw * hilbert_frequency_rms(signal, fs, nfft)), data.size)

    signal = interpolate(signal, data.size) * envelope

    hp_filter = pyloudnorm.IIRfilter(0.0, 0.5, 80.0, fs, 'high_pass')
    meter = pyloudnorm.Meter(fs, 'DeMan')
    meter._filters.__setitem__('hp_filter', hp_filter)
    loudness = meter.integrated_loudness(signal)
    signal = pyloudnorm.normalize.loudness(signal, loudness, -6.0)

    return signal


def process(pattern: str, weights: list = None, nfft: int = 8192, envelope_weight: float = 1.):
    if weights is None:
        weights = [1., .5, .33, .25, .165]
    elif type(weights) is list:
        weights = list(map(lambda x: min(1., float(x)), weights))

    files = sorted(glob(pattern))
    if len(files):
        print(f'processing {len(files)} files..')

        for file in files:
            if 'HearBoost' in file:
                continue
            try:
                print(f'> processing "{file}"..')
                result = process_file(file, weights, nfft, envelope_weight)
                if result is not None:
                    old, new = result
                    print(f'> wrote "{new}"\n')
            except Exception as e:
                print(e)

        print('done.')
    else:
        print('no files to process')
