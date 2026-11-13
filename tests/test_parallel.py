import numpy as np
import pytest

from modulab.config import Config
from modulab import representations as R
from modulab.data.synth import formant_vowel


@pytest.fixture
def cfg():
    return Config.load("modulab_config.yaml")


def test_parallel_flops_is_sum(cfg):
    r_par = R.build("parallel", cfg)
    r_a = R.build("filterbank_amfm", cfg)
    r_b = R.build("stft", cfg)
    n = 16000
    assert r_par.flops_encode(n) == r_a.flops_encode(n) + r_b.flops_encode(n)


def test_parallel_roundtrip(cfg):
    r = R.build("parallel", cfg)
    x = formant_vowel(16000, 1.0, seed=0)
    y = r.decode(r.encode(x, 16000))
    assert len(y) == len(x)
    assert np.isfinite(y).all()


def test_crossover_preserves_tones(cfg):
    cfg.raw["representations"]["parallel"]["fusion"] = "crossover"
    r = R.build("parallel", cfg)
    fs = 16000
    t = np.arange(fs) / fs
    x = (0.3 * np.sin(2 * np.pi * 500 * t) + 0.3 * np.sin(2 * np.pi * 3000 * t)).astype(np.float32)
    y = r.decode(r.encode(x, fs))
    Y = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(len(y), 1.0 / fs)
    e_low = Y[np.argmin(np.abs(freqs - 500))]
    e_high = Y[np.argmin(np.abs(freqs - 3000))]
    assert e_low > 5 and e_high > 5
