from __future__ import annotations
from abc import ABC, abstractmethod


class Operator(ABC):
    name = "base"

    @abstractmethod
    def apply(self, enc): ...

    @abstractmethod
    def compatible(self, repr_name: str) -> bool: ...

    def info(self):
        return {"name": self.name}
