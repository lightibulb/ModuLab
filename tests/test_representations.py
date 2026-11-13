import numpy as np
import pytest

from modulab.config import Config
from modulab import representations as R
from modulab.data.synth import formant_vowel, am_fm_tone


@pytest.fixture
def cfg():
    return Config.load("modulab_config.yaml")


@pytest.mark.parametrize("name,min_snr", [("stft", 15.0), ("filterbank_amfm", 5.0)])
def test_roundtrip_snr(cfg, name, min_snr):
    fs = cfg.fs
    x = formant_vowel(fs, 1.0, seed=0)
    r = R.build(name, cfg)
    enc = r.encode(x, fs)
    y = r.decode(enc)
    n = min(len(x), len(y))
    noise = x[:n] - y[:n]
    snr = 10 * np.log10((np.sum(x[:n] ** 2) + 1e-20) / (np.sum(noise ** 2) + 1e-20))
    assert snr > min_snr, f"{name} SNR={snr:.1f}"


def test_narrowband_amfm(cfg):
    fs = cfg.fs
    x = am_fm_tone(fs, 1.0, fc=1000, am_freq=5, am_depth=0.5, fm_freq=10, fm_depth=10)
    r = R.build("filterbank_amfm", cfg)
    y = r.decode(r.encode(x, fs))
    n = min(len(x), len(y))
    noise = x[:n] - y[:n]
    snr = 10 * np.log10((np.sum(x[:n] ** 2) + 1e-20) / (np.sum(noise ** 2) + 1e-20))
    assert snr > 3.0, f"narrowband AM/FM SNR={snr:.1f}"
