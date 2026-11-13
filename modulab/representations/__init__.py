from .base import Representation, Encoding
from .stft import STFTRepresentation
from .filterbank_amfm import FilterbankAMFM
from .parallel import ParallelRepresentation

REGISTRY = {
    "stft": STFTRepresentation,
    "filterbank_amfm": FilterbankAMFM,
    "parallel": ParallelRepresentation,
}


def build(name, cfg):
    sub = cfg.get("representations", name, default={}) or {}
    if name == "stft":
        return STFTRepresentation(n_fft=sub.get("n_fft", 512),
                                  hop_length=sub.get("hop_length", 128),
                                  window=sub.get("window", "hann"))
    if name == "filterbank_amfm":
        return FilterbankAMFM(num_bands=sub.get("num_bands", 24),
                              f_min=sub.get("f_min_hz", 100),
                              f_max=sub.get("f_max_hz", 7000),
                              order=sub.get("filter_order", 4),
                              spacing=sub.get("spacing", "erb"))
    if name == "parallel":
        branches = sub.get("branches", ["filterbank_amfm", "stft"])
        return ParallelRepresentation(
            sub_reps=[build(b, cfg) for b in branches],
            fusion=sub.get("fusion", "average"),
            weights=sub.get("weights"),
            crossover_hz=sub.get("crossover_hz", 1500.0),
            low_from=sub.get("low_from", 0),
            high_from=sub.get("high_from", 1),
        )
    raise ValueError(f"Unknown representation: {name}")
