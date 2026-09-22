"""Filters run between the engine and the outputs."""
from __future__ import annotations

import time
from pathlib import Path

from thai_alpr.core.plugin import FilterPlugin, register
from thai_alpr.core.types import PlateResult


@register("filter", "min_conf")
class MinConfFilter(FilterPlugin):
    """Drop readings below a threshold. config: threshold (0.3)"""

    def apply(self, r: PlateResult):
        return r if r.conf >= float(self.config.get("threshold", 0.3)) else None


@register("filter", "dedupe")
class DedupeFilter(FilterPlugin):
    """Suppress the same plate from the same source within `seconds` (default 10)."""

    def start(self):
        self.seen: dict[tuple[str, str], float] = {}

    def apply(self, r: PlateResult):
        key = (r.source, r.plate)
        now = time.time()
        if now - self.seen.get(key, 0) < float(self.config.get("seconds", 10)):
            return None
        self.seen[key] = now
        return r


@register("filter", "allowlist")
class AllowlistFilter(FilterPlugin):
    """Mark r.allowed = True when the plate is in a text file (one plate per line, e.g. "1กข 1234").
    Outputs like mqtt/serial use that flag to open a gate. config: path, block_others (false)"""

    def start(self):
        self.path = Path(self.config.get("path", "configs/allowlist.txt"))
        self._mtime = 0.0
        self._load()

    def _load(self):
        if self.path.exists() and self.path.stat().st_mtime != self._mtime:
            self._mtime = self.path.stat().st_mtime
            self.plates = {ln.strip().replace(" ", "") for ln in self.path.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.startswith("#")}
        elif not self.path.exists():
            self.plates = set()

    def apply(self, r: PlateResult):
        self._load()
        r.allowed = r.plate.replace(" ", "") in self.plates      # type: ignore[attr-defined]
        if self.config.get("block_others") and not r.allowed:
            return None
        return r
