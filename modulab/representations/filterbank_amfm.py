from __future__ import annotations
import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert

from .base import Representation, Encoding


def erb_edges(num_bands, f_min, f_max):
    erb_min = 21.4 * np.log10(1 + 0.00437 * f_min)
    erb_max = 21.4 * np.log10(1 + 0.00437 * f_max)
    pts = np.linspace(erb_min, erb_max, num_bands)
    centers = (10 ** (pts / 21.4) - 1) / 0.00437
    edges = np.zeros(num_bands + 1)
    edges[0] = f_min
    edges[-1] = f_max
    for k in range(1, num_bands):
        edges[k] = np.sqrt(centers[k - 1] * centers[k])
    return edges, centers


class FilterbankAMFM(Representation):
    name = "filterbank_amfm"

    def __init__(self, num_bands=24, f_min=100.0, f_max=7000.0, order=4, spacing="erb"):
        self.num_bands = num_bands
        self.f_min = f_min
        self.f_max = f_max
        self.order = order
        self.spacing = spacing

    def encode(self, x, fs):
        edges, centers = erb_edges(self.num_bands, self.f_min, self.f_max)
        nyq = fs / 2.0
        n = len(x)
        am = np.zeros((self.num_bands, n), dtype=np.float32)
        fm = np.zeros((self.num_bands, n), dtype=np.float32)
        phase0 = np.zeros(self.num_bands, dtype=np.float32)

        for k in range(self.num_bands):
            lo = edges[k] / nyq
            hi = min(edges[k + 1] / nyq, 0.999)
            if hi <= lo:
                continue
            sos = butter(self.order, [lo, hi], btype="band", output="sos")
            xk = sosfiltfilt(sos, x)
            analytic = hilbert(xk)
            am[k] = np.abs(analytic).astype(np.float32)
            phase = np.unwrap(np.angle(analytic))
            carrier = 2 * np.pi * centers[k] * np.arange(n) / fs
            residual = phase - carrier
            # CRITICAL: preserve initial phase per band before zeroing.
            phase0[k] = float(residual[0])
            residual = residual - residual[0]
            # First-difference with leading zero -> cumsum exactly inverts.
            fm_k = np.zeros(n, dtype=np.float32)
            fm_k[1:] = (np.diff(residual) * fs / (2 * np.pi)).astype(np.float32)
            fm[k] = fm_k

        return Encoding(
            data={"am": am, "fm": fm, "phase0": phase0,
                  "centers": centers, "edges": edges},
            meta={"n_samples": n, "fs": fs, "num_bands": self.num_bands},
        )

    def decode(self, enc):
        am = enc.data["am"]
        fm = enc.data["fm"]
        phase0 = enc.data["phase0"]
        centers = enc.data["centers"]
        fs = enc.meta["fs"]
        n = enc.meta["n_samples"]
        t = np.arange(n) / fs
        y = np.zeros(n, dtype=np.float64)
        for k, fc in enumerate(centers):
            carrier = 2 * np.pi * fc * t
            rec_residual = 2 * np.pi * np.cumsum(fm[k].astype(np.float64)) / fs
            phase = phase0[k] + carrier + rec_residual
            y += am[k] * np.cos(phase)
        return y.astype(np.float32)

    def quantize(self, enc, bits):
        am = enc.data["am"].copy()
        fm = enc.data["fm"].copy()
        am_max = float(am.max()) + 1e-12
        am_lv = 2 ** bits - 1
        am_q = np.round(am / am_max * am_lv) / am_lv * am_max
        fm_max = float(np.abs(fm).max()) + 1e-12
        fm_lv = 2 ** (bits - 1) - 1
        fm_q = np.round(fm / fm_max * fm_lv) / fm_lv * fm_max
        return Encoding(
            data={"am": am_q.astype(np.float32), "fm": fm_q.astype(np.float32),
                  "phase0": enc.data["phase0"],
                  "centers": enc.data["centers"], "edges": enc.data["edges"]},
            meta=dict(enc.meta),
        )

    def flops_encode(self, n_samples):
        fft = 5 * n_samples * int(np.log2(max(n_samples, 2)))
        iir = 20 * n_samples
        return self.num_bands * (fft + iir)

    def flops_decode(self, n_samples):
        return self.num_bands * 4 * n_samples

    def info(self):
        edges, centers = erb_edges(self.num_bands, self.f_min, self.f_max)
        ratios = np.diff(edges) / centers
        return {
            "name": self.name, "num_bands": self.num_bands,
            "f_min_hz": self.f_min, "f_max_hz": self.f_max,
            "filter_order": self.order,
            "bw_fc_ratio_min": float(ratios.min()),
            "bw_fc_ratio_max": float(ratios.max()),
            "bw_fc_ratio_median": float(np.median(ratios)),
        }
