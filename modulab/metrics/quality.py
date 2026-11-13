from __future__ import annotations
import numpy as np
from scipy.signal import stft as sp_stft
from scipy.fftpack import dct


def snr_db(x_ref, x_out):
    x_ref = np.asarray(x_ref, dtype=np.float64)
    x_out = np.asarray(x_out, dtype=np.float64)
    n = min(len(x_ref), len(x_out))
    x_ref, x_out = x_ref[:n], x_out[:n]
    noise = x_ref - x_out
    return float(10 * np.log10((np.sum(x_ref ** 2) + 1e-20) / (np.sum(noise ** 2) + 1e-20)))


def si_sdr_db(x_ref, x_out):
    x_ref = np.asarray(x_ref, dtype=np.float64)
    x_out = np.asarray(x_out, dtype=np.float64)
    n = min(len(x_ref), len(x_out))
    x_ref, x_out = x_ref[:n], x_out[:n]
    x_ref = x_ref - x_ref.mean()
    x_out = x_out - x_out.mean()
    a = np.dot(x_out, x_ref) / (np.dot(x_ref, x_ref) + 1e-20)
    tgt = a * x_ref
    noise = x_out - tgt
    return float(10 * np.log10((np.sum(tgt ** 2) + 1e-20) / (np.sum(noise ** 2) + 1e-20)))


def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10 ** (m / 2595.0) - 1.0)


def _mel_filterbank(fs, n_fft, n_mels=26, f_min=0.0, f_max=None):
    if f_max is None:
        f_max = fs / 2.0
    mels = np.linspace(_hz_to_mel(f_min), _hz_to_mel(f_max), n_mels + 2)
    hz = _mel_to_hz(mels)
    freqs = np.linspace(0, fs / 2.0, n_fft // 2 + 1)
    fb = np.zeros((n_mels, len(freqs)))
    for i in range(n_mels):
        lo, ctr, hi = hz[i], hz[i + 1], hz[i + 2]
        for j, f in enumerate(freqs):
            if lo <= f <= ctr:
                fb[i, j] = (f - lo) / (ctr - lo + 1e-12)
            elif ctr < f <= hi:
                fb[i, j] = (hi - f) / (hi - ctr + 1e-12)
    return fb


def _mfcc(x, fs, n_mfcc=13, n_mels=26, n_fft=512):
    _, _, Z = sp_stft(x.astype(np.float64), fs=fs, nperseg=n_fft)
    mag = np.abs(Z)
    fb = _mel_filterbank(fs, n_fft, n_mels)
    mel = fb @ mag
    log_mel = np.log(mel + 1e-10)
    return dct(log_mel, type=2, axis=0, norm="ortho")[:n_mfcc]


def mcd_db(x_ref, x_out, fs=16000, n_mfcc=13):
    try:
        n = min(len(x_ref), len(x_out))
        m1 = _mfcc(np.asarray(x_ref[:n], dtype=np.float32), fs, n_mfcc)
        m2 = _mfcc(np.asarray(x_out[:n], dtype=np.float32), fs, n_mfcc)
        diff = m1[1:] - m2[1:]
        return float(np.mean(np.sqrt(np.sum(diff ** 2, axis=0))) * (10.0 / np.log(10)) * np.sqrt(2.0))
    except Exception:
        return float("nan")
