from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class Encoding:
    data: dict[str, Any]
    meta: dict[str, Any]


class Representation(ABC):
    name: str = "base"

    @abstractmethod
    def encode(self, x: np.ndarray, fs: int) -> Encoding: ...

    @abstractmethod
    def decode(self, enc: Encoding) -> np.ndarray: ...

    @abstractmethod
    def quantize(self, enc: Encoding, bits: int) -> Encoding: ...

    @abstractmethod
    def flops_encode(self, n_samples: int) -> int: ...

    @abstractmethod
    def flops_decode(self, n_samples: int) -> int: ...

    def apply_operator(self, enc: Encoding, op) -> Encoding:
        return op.apply(enc)

    def info(self) -> dict[str, Any]:
        return {"name": self.name}
