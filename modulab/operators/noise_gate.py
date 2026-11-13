import numpy as np

from modulab.operators.base import Operator
from modulab.representations.base import Encoding


class NoiseGate(Operator):
    name = "noise_gate"

    def __init__(self, threshold_db=-40.0):
        self.threshold_db = threshold_db
        self.threshold_linear = 10 ** (threshold_db / 20.0)

    def apply(self, enc):
        if "am" in enc.data:
            am = enc.data["am"].copy()
            am[am < self.threshold_linear * am.max()] = 0.0
            return Encoding(data={**enc.data, "am": am}, meta=dict(enc.meta))
        if "mag" in enc.data:
            mag = enc.data["mag"].copy()
            mag[mag < self.threshold_linear * mag.max()] = 0.0
            return Encoding(data={**enc.data, "mag": mag}, meta=dict(enc.meta))
        return enc

    def compatible(self, repr_name):
        return repr_name in {"stft", "filterbank_amfm"}
