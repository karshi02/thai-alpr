"""Plate type from background colour + per-track majority vote."""
from __future__ import annotations

from collections import Counter, defaultdict, deque

import cv2
import numpy as np


def plate_type_from_color(crop: np.ndarray) -> str:
    """Rough HSV classification of the plate background.
    white  -> private car        yellow -> taxi / hired
    green  -> commercial truck   red    -> temporary (ป้ายแดง)
    Anything with a very non-uniform background -> graphic (auction plate)."""
    if crop.size == 0:
        return "unknown"
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    # sample the border region (chars sit in the middle)
    h, w = hsv.shape[:2]
    b = max(2, int(min(h, w) * 0.08))
    border = np.concatenate([hsv[:b].reshape(-1, 3), hsv[-b:].reshape(-1, 3), hsv[:, :b].reshape(-1, 3), hsv[:, -b:].reshape(-1, 3)])
    hue, sat, val = border[:, 0].astype(int), border[:, 1].astype(int), border[:, 2].astype(int)
    if sat.std() > 55 or hue.std() > 40:
        return "graphic"
    s, v, hm = np.median(sat), np.median(val), np.median(hue)
    if s < 50 and v > 140:
        return "private"
    if 15 <= hm <= 40 and s > 80:
        return "taxi"
    if 40 < hm <= 90 and s > 60:
        return "commercial"
    if (hm < 10 or hm > 165) and s > 90:
        return "temporary"
    return "unknown"


class TrackVoter:
    """Keep the last N readings per track id and return the majority plate/province.
    Confidence returned = fraction of votes for the winner * mean conf."""

    def __init__(self, window: int = 15, min_votes: int = 3):
        self.window = window
        self.min_votes = min_votes
        self._hist: dict[int, deque] = defaultdict(lambda: deque(maxlen=window))
        self.emitted: dict[int, str] = {}

    def vote(self, track_id: int, plate: str, province: str, conf: float) -> tuple[str, str, float, bool]:
        """Returns (plate, province, conf, is_new) - is_new True the first time a track settles."""
        if plate:
            self._hist[track_id].append((plate, province, conf))
        hist = self._hist[track_id]
        if len(hist) < self.min_votes:
            return plate, province, conf, False
        plates = Counter(p for p, _, _ in hist)
        best, n = plates.most_common(1)[0]
        provs = Counter(v for p, v, _ in hist if p == best and v)
        prov = provs.most_common(1)[0][0] if provs else ""
        mean_conf = float(np.mean([c for p, _, c in hist if p == best]))
        conf = (n / len(hist)) * mean_conf
        is_new = self.emitted.get(track_id) != best
        if is_new:
            self.emitted[track_id] = best
        return best, prov, conf, is_new

    def forget(self, active_ids: set[int]) -> None:
        for tid in list(self._hist):
            if tid not in active_ids:
                self._hist.pop(tid, None)
                self.emitted.pop(tid, None)
