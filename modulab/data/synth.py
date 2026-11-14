from __future__ import annotations
import numpy as np


def formant_vowel(fs, duration_s, f0=120.0, formants=(700, 1220, 2600), seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(fs * duration_s)) / fs
    f0_inst = f0 + 5 * np.sin(2 * np.pi * 0.5 * t) + 2 * rng.standard_normal(len(t)).cumsum() * 0.001
    phase = 2 * np.pi * np.cumsum(f0_inst) / fs
    y = np.zeros_like(t)
    for h in range(1, 31):
        fh = h * f0
        if fh > fs / 2 - 200:
            break
        amp = 0.0
        for F in formants:
            bw = 80.0
            amp += 1.0 / (1.0 + ((fh - F) / (bw / 2)) ** 2)
        y += amp * np.sin(h * phase + rng.uniform(0, 2 * np.pi))
    y /= (np.abs(y).max() + 1e-12)
    env = np.ones_like(t)
    for start in [0.1, 0.9, 1.7, 2.4]:
        if start < duration_s:
            env *= 1.0 - 0.5 * np.exp(-((t - start) ** 2) / 0.02)
    return (y * env).astype(np.float32)


def am_fm_tone(fs, duration_s, fc, am_freq=5.0, am_depth=0.5,
               fm_freq=10.0, fm_depth=20.0, seed=0):
    t = np.arange(int(fs * duration_s)) / fs
    env = 1.0 + am_depth * np.cos(2 * np.pi * am_freq * t)
    phase = 2 * np.pi * fc * t + (fm_depth / fm_freq) * np.sin(2 * np.pi * fm_freq * t)
    return (env * np.cos(phase)).astype(np.float32)


def generate_synthetic_corpus(fs, duration_s, seed=42):
    """Synthetic corpus. `seed` offsets the internal RNG seeds so multi-seed
    benchmark runs get genuinely different signals."""
    base = int(seed) * 1000
    return {
        "vowel_a": formant_vowel(fs, duration_s, f0=120, formants=(730, 1090, 2440), seed=base + 1),
        "vowel_i": formant_vowel(fs, duration_s, f0=140, formants=(270, 2290, 3010), seed=base + 2),
        "vowel_u": formant_vowel(fs, duration_s, f0=100, formants=(300, 870, 2240), seed=base + 3),
        "am_tone_1k": am_fm_tone(fs, duration_s, fc=1000, am_freq=5, am_depth=0.5,
                                 fm_freq=10, fm_depth=10, seed=base + 4),
        "fm_tone_1k": am_fm_tone(fs, duration_s, fc=1000, am_freq=0.5, am_depth=0.1,
                                 fm_freq=20, fm_depth=50, seed=base + 5),
        "amfm_tone_2k": am_fm_tone(fs, duration_s, fc=2000, am_freq=4, am_depth=0.6,
                                   fm_freq=8, fm_depth=30, seed=base + 6),
    }
