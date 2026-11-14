"""Empirical spectral analysis helpers — measure actual spectral support
rather than relying on analytical bounds. Used by Bedrosian tests and
filterbank coverage diagnostics."""
from __future__ import annotations
import numpy as np


def spectral_upper_edge(sig, fs, thresh_db=-40.0):
    """Highest frequency (Hz) at which the magnitude spectrum exceeds
    thresh_db below peak."""
    S = np.abs(np.fft.rfft(sig))
    freqs = np.fft.rfftfreq(len(sig), 1.0 / fs)
    if S.max() <= 0:
        return 0.0
    thresh = S.max() * 10 ** (thresh_db / 20.0)
    above = freqs[S >= thresh]
    return float(above.max()) if len(above) else 0.0


def spectral_lower_edge(sig, fs, thresh_db=-40.0, exclude_dc_hz=5.0):
    """Lowest non-DC frequency (Hz) above thresh_db below peak."""
    S = np.abs(np.fft.rfft(sig))
    freqs = np.fft.rfftfreq(len(sig), 1.0 / fs)
    if S.max() <= 0:
        return 0.0
    thresh = S.max() * 10 ** (thresh_db / 20.0)
    mask = (freqs > exclude_dc_hz) & (S >= thresh)
    above = freqs[mask]
    return float(above.min()) if len(above) else 0.0


def filterbank_coverage_snr(sos_list, x, fs):
    """Diagnostic: how well does sum of all band filters reconstruct x?
    A perfect-reconstruction filterbank gives high SNR (>40 dB); a filterbank
    with spectral gaps gives lower SNR — exposing the reconstruction ceiling."""
    from scipy.signal import sosfiltfilt
    y = np.zeros_like(x, dtype=np.float64)
    for sos in sos_list:
        y += sosfiltfilt(sos, x)
    noise = x - y
    return 10 * np.log10((np.sum(x ** 2) + 1e-20) / (np.sum(noise ** 2) + 1e-20))
