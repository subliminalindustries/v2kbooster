# coding: utf-8

import audiofile
import librosa
import numpy as np
import os
import pyloudnorm
import sys
import traceback
import warnings

from glob import glob
from scipy.signal import hilbert, savgol_filter
from sklearn.preprocessing import minmax_scale
from soundfile import LibsndfileError

warnings.filterwarnings('ignore', category=UserWarning)

MAX_CHUNK_SIZE = 50000000


def load_file(filename: str,
              mono: bool = False) -> (np.ndarray, int):
    try:
        signal, fs = audiofile.read(filename)
        if mono and len(signal.shape) == 2:
            signal = signal[0]
    except LibsndfileError:
        signal, fs = librosa.load(filename, mono=mono)

    return signal, fs


def interpolate(data: np.ndarray, to_size: int) -> np.ndarray:
    return np.interp(np.linspace(0, data.size - 1, num=to_size), np.arange(data.size), data)


def hilbert_frequency_rms(data: np.ndarray,
                          fs: int,
                          window_size: int = 16) -> np.ndarray:
    w = min(64, max(16, window_size // 32))
    f = (np.diff(np.unwrap(np.angle(hilbert(data)))) / (2.0 * np.pi) * fs)

    mf = []
    for x in range(0, f.size - w, w):
        mf.append(np.sqrt(np.mean(f[x:x + w]) ** 2))

    return 1-savgol_filter(np.array(mf), w, 1)


def enhance_harmonic_saliency(data: np.ndarray,
                              fs: int,
                              harmonics: list,
                              weights: list,
                              nfft: int,
                              hw: float) -> np.ndarray:
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

    return signal


def process_file(filename: str,
                 weights: list,
                 nfft: int,
                 hw: float,
                 overwrite: bool) -> bool:
    if filename.find('-enhanced.wav') != -1:
        return False

    path, ext = os.path.splitext(filename)
    dir_name = os.path.dirname(path)
    if os.path.islink(dir_name):
        dir_name = os.path.realpath(dir_name)
    new_filename = os.path.join(dir_name, f'{os.path.basename(path)}-enhanced{ext}')

    if os.path.isfile(new_filename) or os.path.islink(new_filename):
        if overwrite:
            os.remove(new_filename)
        else:
            return False

    print(f'> processing "{filename}"..')

    harmonics = list(np.arange(1, len(weights)+1))
    harmonic_weights = ', '.join(list(map(lambda w: f'{w[0]}={w[1]}', zip(harmonics, weights))))

    print(f'harmonic weights: {harmonic_weights}')
    print(f'fft bins: {nfft}')

    signal, fs = load_file(filename, mono=True)
    print(f'samples: {len(signal)}')

    meter = pyloudnorm.Meter(fs)
    loudness = meter.integrated_loudness(signal)
    signal = pyloudnorm.normalize.loudness(signal, loudness, -30.0)
    signal = np.nan_to_num(signal, posinf=1., neginf=-1.)

    samples = len(signal)
    chunks = len(signal) // MAX_CHUNK_SIZE
    samples -= (chunks * MAX_CHUNK_SIZE)
    if samples:
        chunks += 1

    parts = []
    for x in range(0, chunks):
        rem = min(len(signal), MAX_CHUNK_SIZE)
        if rem:
            parts.append(signal[:rem-1])
            signal = signal[rem:]

    if len(parts) > 1:
        print(f'! large file, broken up into {len(parts)} chunks to prevent resource issues.')

    for k, part in enumerate(parts):
        if len(parts) > 1:
            print(f'> chunk {k+1}..', end='\r')
        parts[k] = minmax_scale(enhance_harmonic_saliency(part, fs, harmonics, weights, nfft, hw), (-1., 1.))

    enhanced = np.block(parts)

    hp_filter = pyloudnorm.IIRfilter(0.0, 0.7, 80.0, fs, 'high_pass')
    meter = pyloudnorm.Meter(fs, 'DeMan')
    meter._filters.__setitem__('hp_filter', hp_filter)
    loudness = meter.integrated_loudness(enhanced)
    enhanced = pyloudnorm.normalize.loudness(enhanced, loudness, -20.0)

    audiofile.write(file=new_filename, signal=enhanced, sampling_rate=fs)

    print(f'> wrote "{new_filename}"\n{"-" * 80}')

    return True


def process(pattern: str,
            weights: list = None,
            nfft: int = 8192,
            envelope_weight: float = 1.,
            overwrite: bool = False) -> None:
    if weights is None:
        weights = [1., .5, .33, .25, .165]
    elif type(weights) is list:
        weights = list(map(lambda x: min(1., float(x)), weights))

    files = sorted(glob(pattern, recursive=True))
    print(f'found {len(files)} files.')

    done_work = False

    for file in files:
        try:
            if process_file(file, weights, nfft, envelope_weight, overwrite) and done_work is False:
                done_work = True
        except Exception as e:
            print(''.join(traceback.format_exception(sys.exc_info()[0], sys.exc_info()[1], sys.exc_info()[2])))

    if done_work:
        print('done.\n')
    else:
        print('nothing to do.\n')
