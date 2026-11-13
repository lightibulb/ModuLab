from __future__ import annotations
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any]
    hash: str

    @classmethod
    def load(cls, path: str | Path = "modulab_config.yaml") -> "Config":
        text = Path(path).read_text()
        raw = yaml.safe_load(text)
        h = hashlib.sha256(text.encode()).hexdigest()[:12]
        return cls(raw=raw, hash=h)

    def get(self, *keys: str, default: Any = None) -> Any:
        cur: Any = self.raw
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur

    @property
    def fs(self) -> int:
        return int(self.raw["audio"]["fs"])

    @property
    def duration_s(self) -> float:
        return float(self.raw["audio"]["duration_s"])
