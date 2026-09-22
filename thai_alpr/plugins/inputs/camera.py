"""Inputs: webcam / video file / RTSP (all through OpenCV) and image folder."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Iterator

import cv2

from thai_alpr.core.plugin import InputPlugin, register
from thai_alpr.core.types import Frame


@register("input", "camera")
class CameraInput(InputPlugin):
    """config: source (int index | file path | rtsp:// url), fps_limit, loop"""

    def start(self):
        src = self.config.get("source", 0)
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open {src}")
        self.min_dt = 1.0 / float(self.config.get("fps_limit", 30))

    def frames(self) -> Iterator[Frame]:
        last = 0.0
        while True:
            ok, img = self.cap.read()
            if not ok:
                if self.config.get("loop") and isinstance(self.config.get("source"), str):
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                return
            now = time.time()
            if now - last < self.min_dt:
                continue
            last = now
            yield Frame(image=img, source=self.label, ts=now)

    def stop(self):
        if getattr(self, "cap", None):
            self.cap.release()


@register("input", "rtsp")
class RtspInput(CameraInput):
    """Alias of camera with a url key. config: url, fps_limit"""

    def start(self):
        self.config["source"] = self.config["url"]
        super().start()


@register("input", "folder")
class FolderInput(InputPlugin):
    """Read every image in a folder once. config: path, glob (default *.jpg)"""

    def frames(self) -> Iterator[Frame]:
        root = Path(self.config["path"])
        patterns = self.config.get("glob", "*.jpg,*.jpeg,*.png").split(",")
        files = sorted(p for pat in patterns for p in root.glob(pat.strip()))
        for p in files:
            img = cv2.imread(str(p))
            if img is None:
                continue
            yield Frame(image=img, source=self.label, meta={"file": str(p), "still": True})
