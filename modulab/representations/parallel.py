from __future__ import annotations
import numpy as np

from .base import Representation, Encoding


class ParallelRepresentation(Representation):
    name = "parallel"

    def __init__(self, sub_reps, fusion="average", weights=None,
                 crossover_hz=1500.0, low_from=0, high_from=1):
        if fusion not in ("average", "crossover"):
            raise ValueError(f"Unknown fusion: {fusion}")
        self.sub_reps = sub_reps
        self.fusion = fusion
        self.weights = weights or [1.0 / len(sub_reps)] * len(sub_reps)
        self.crossover_hz = crossover_hz
        self.low_from = low_from
        self.high_from = high_from

    def encode(self, x, fs):
        branch_encs = [r.encode(x, fs) for r in self.sub_reps]
        return Encoding(
            data={"branches": branch_encs},
            meta={"n_samples": len(x), "fs": fs,
                  "branch_names": [r.name for r in self.sub_reps]},
        )

    def decode(self, enc):
        fs = enc.meta["fs"]
        n = enc.meta["n_samples"]
        ys = [np.asarray(r.decode(e)[:n], dtype=np.float64)
              for r, e in zip(self.sub_reps, enc.data["branches"])]
        if self.fusion == "average":
            out = sum(w * y for w, y in zip(self.weights, ys))
            return out.astype(np.float32)
        Ys = [np.fft.rfft(y, n=n) for y in ys]
        freqs = np.fft.rfftfreq(n, 1.0 / fs)
        mask_low = freqs < self.crossover_hz
        Y = np.zeros_like(Ys[0])
        Y[mask_low] = Ys[self.low_from][mask_low]
        Y[~mask_low] = Ys[self.high_from][~mask_low]
        return np.fft.irfft(Y, n=n).astype(np.float32)

    def quantize(self, enc, bits):
        branch_encs = [r.quantize(e, bits)
                       for r, e in zip(self.sub_reps, enc.data["branches"])]
        return Encoding(data={"branches": branch_encs}, meta=dict(enc.meta))

    def apply_operator(self, enc, op):
        out = []
        for r, e in zip(self.sub_reps, enc.data["branches"]):
            out.append(op.apply(e) if op.compatible(r.name) else e)
        return Encoding(data={"branches": out}, meta=dict(enc.meta))

    def flops_encode(self, n_samples):
        return sum(r.flops_encode(n_samples) for r in self.sub_reps)

    def flops_decode(self, n_samples):
        sub = sum(r.flops_decode(n_samples) for r in self.sub_reps)
        if self.fusion == "average":
            fusion = len(self.sub_reps) * n_samples
        else:
            fusion = 3 * 5 * n_samples * int(np.log2(max(n_samples, 2))) + n_samples
        return sub + fusion

    def info(self):
        return {"name": self.name, "fusion": self.fusion,
                "weights": list(self.weights), "crossover_hz": self.crossover_hz,
                "branches": [r.info() for r in self.sub_reps]}
