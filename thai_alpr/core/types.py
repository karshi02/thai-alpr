"""Shared data types passed between plugins and engine stages."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class Frame:
    """One image coming from an input plugin."""
    image: np.ndarray                 # BGR, HxWx3
    source: str                       # plugin name, e.g. "esp32cam:gate1"
    ts: float = field(default_factory=time.time)
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class PlateResult:
    """Final reading of one plate in one frame."""
    plate: str                        # e.g. "1กข 1234"  ("" if unreadable)
    province: str                     # e.g. "กรุงเทพมหานคร" ("" if unknown)
    plate_type: str                   # private | taxi | commercial | temporary | graphic | unknown
    conf: float                       # combined confidence 0-1
    bbox: tuple[int, int, int, int]   # x1, y1, x2, y2 on the original frame
    source: str
    ts: float
    track_id: Optional[int] = None
    raw_chars: list[dict] = field(default_factory=list)   # per-char {char, conf, x}
    plate_crop: Optional[np.ndarray] = None               # not serialised

    def to_dict(self) -> dict[str, Any]:
        return {
            "plate": self.plate,
            "province": self.province,
            "plate_type": self.plate_type,
            "conf": round(self.conf, 3),
            "bbox": list(self.bbox),
            "source": self.source,
            "ts": self.ts,
            "track_id": self.track_id,
        }
