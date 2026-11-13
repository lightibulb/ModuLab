from .base import Operator
from .passthrough import Passthrough
from .noise_gate import NoiseGate

REGISTRY = {"passthrough": Passthrough, "noise_gate": NoiseGate}


def build(name):
    if name == "passthrough":
        return Passthrough()
    if name == "noise_gate":
        return NoiseGate(threshold_db=-40.0)
    raise ValueError(f"Unknown operator: {name}")
