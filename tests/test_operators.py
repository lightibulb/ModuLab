import numpy as np
from modulab.config import Config
from modulab import representations as R, operators as O


def test_passthrough_identity():
    cfg = Config.load("modulab_config.yaml")
    x = np.random.randn(16000).astype(np.float32) * 0.1
    r = R.build("stft", cfg)
    enc = r.encode(x, cfg.fs)
    y1 = r.decode(enc)
    y2 = r.decode(O.build("passthrough").apply(enc))
    np.testing.assert_allclose(y1, y2, atol=1e-5)


def test_noise_gate_amfm():
    cfg = Config.load("modulab_config.yaml")
    x = np.random.randn(16000).astype(np.float32) * 0.001
    r = R.build("filterbank_amfm", cfg)
    enc = r.encode(x, cfg.fs)
    enc2 = O.build("noise_gate").apply(enc)
    assert np.any(enc2.data["am"] == 0.0)
