import numpy as np
from modulab.metrics.quality import snr_db, si_sdr_db, mcd_db


def test_snr_identical():
    x = np.random.randn(1000)
    assert snr_db(x, x) > 100


def test_si_sdr_scale():
    x = np.random.randn(1000)
    assert si_sdr_db(x, 3.0 * x) > 100


def test_mcd_identical():
    x = np.random.randn(16000).astype(np.float32) * 0.1
    assert mcd_db(x, x) < 0.01
