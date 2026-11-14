"""Equal-bitrate comparison: compute bits/second of each representation."""
from __future__ import annotations


def bits_per_second(repr_obj, n_samples, bits, fs=16000):
    """Total bits per second of the encoded representation.

    STFT:  n_frames * n_bins * 2 (mag+phase) * bits / duration
    AM/FM: num_bands * 2 (AM+FM) * n_samples * bits / duration
    """
    if n_samples <= 0:
        return float("nan")
    duration_s = n_samples / fs
    name = getattr(repr_obj, "name", "unknown")

    if name == "stft":
        n_frames = 1 + n_samples // repr_obj.hop
        n_bins = repr_obj.n_fft // 2 + 1
        return (n_frames * n_bins * 2 * bits) / duration_s

    if name == "filterbank_amfm":
        return (repr_obj.num_bands * 2 * n_samples * bits) / duration_s

    if name == "parallel":
        return sum(bits_per_second(r, n_samples, bits, fs)
                   for r in repr_obj.sub_reps)

    return float("nan")
