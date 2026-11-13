from __future__ import annotations
import numpy as np
from scipy.signal import stft as sp_stft, istft as sp_istft

from .base import Representation, Encoding


class STFTRepresentation(Representation):
    name = "stft"

    def __init__(self, n_fft: int = 512, hop_length: int = 128, window: str = "hann"):
        self.n_fft = n_fft
        self.hop = hop_length
        self.window = window

    def encode(self, x: np.ndarray, fs: int) -> Encoding:
        noverlap = self.n_fft - self.hop
        _, _, Z = sp_stft(
            x, fs=fs, window=self.window, nperseg=self.n_fft,
            noverlap=noverlap, boundary="zeros", padded=True,
        )
        return Encoding(
            data={"mag": np.abs(Z).astype(np.float32),
                  "phase": np.angle(Z).astype(np.float32)},
            meta={"n_samples": len(x), "fs": fs, "n_fft": self.n_fft, "hop": self.hop},
        )

    def decode(self, enc: Encoding) -> np.ndarray:
        mag = enc.data["mag"]
        phase = enc.data["phase"]
        Z = mag * np.exp(1j * phase)
        noverlap = self.n_fft - enc.meta["hop"]
        _, x = sp_istft(
            Z, fs=enc.meta["fs"], window=self.window, nperseg=self.n_fft,
            noverlap=noverlap, input_onesided=True, boundary=True,
        )
        return x[: enc.meta["n_samples"]].astype(np.float32)

    def quantize(self, enc: Encoding, bits: int) -> Encoding:
        levels = 2 ** bits - 1
        mag = enc.data["mag"]
        phase = enc.data["phase"]
        mag_max = float(mag.max()) + 1e-12
        mag_q = np.round(mag / mag_max * levels) / levels * mag_max
        phase_q = np.round((phase + np.pi) / (2 * np.pi) * levels) / levels * 2 * np.pi - np.pi
        return Encoding(
            data={"mag": mag_q.astype(np.float32), "phase": phase_q.astype(np.float32)},
            meta=dict(enc.meta),
        )

    def flops_encode(self, n_samples: int) -> int:
        n_frames = 1 + n_samples // self.hop
        n_bins = self.n_fft // 2 + 1
        return int(n_frames * 5 * self.n_fft * np.log2(self.n_fft) + n_frames * n_bins * 4)

    def flops_decode(self, n_samples: int) -> int:
        return self.flops_encode(n_samples)

    def info(self) -> dict:
        return {"name": self.name, "n_fft": self.n_fft, "hop_length": self.hop}
