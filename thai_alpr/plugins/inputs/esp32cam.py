"""ESP32-CAM input.

Two modes, pick with `mode`:
  snapshot (default) - GET http://<ip>/capture every 1/fps seconds (stock CameraWebServer sketch)
  stream             - MJPEG at http://<ip>:81/stream, decoded with OpenCV

The firmware in firmware/esp32cam_streamer/ serves both endpoints and adds a
`X-Device-Id` header so several boards can share one pipeline.
"""
from __future__ import annotations

import time
from typing import Iterator

import cv2
import numpy as np
import requests

from thai_alpr.core.plugin import InputPlugin, register
from thai_alpr.core.types import Frame


@register("input", "esp32cam")
class Esp32CamInput(InputPlugin):
    def start(self):
        self.host = self.config["host"].rstrip("/")
        if not self.host.startswith("http"):
            self.host = "http://" + self.host
        self.mode = self.config.get("mode", "snapshot")
        self.dt = 1.0 / float(self.config.get("fps", 4))
        self.timeout = float(self.config.get("timeout", 3))
        self.session = requests.Session()

    def _snapshot(self) -> Iterator[Frame]:
        url = self.host + self.config.get("capture_path", "/capture")
        fails = 0
        while True:
            t = time.time()
            try:
                r = self.session.get(url, timeout=self.timeout)
                r.raise_for_status()
                img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
                fails = 0
                if img is not None:
                    yield Frame(image=img, source=self.label, ts=t, meta={"device": r.headers.get("X-Device-Id", self.instance)})
            except Exception as e:
                fails += 1
                if fails in (1, 10, 100):
                    print(f"[esp32cam:{self.instance}] {e}")
                time.sleep(min(5, 0.5 * fails))
            time.sleep(max(0, self.dt - (time.time() - t)))

    def _stream(self) -> Iterator[Frame]:
        url = self.config.get("stream_url") or self.host.replace(":80", "") + ":81/stream"
        while True:
            cap = cv2.VideoCapture(url)
            if not cap.isOpened():
                print(f"[esp32cam:{self.instance}] cannot open {url}, retry in 3s")
                time.sleep(3)
                continue
            last = 0.0
            while True:
                ok, img = cap.read()
                if not ok:
                    break
                now = time.time()
                if now - last < self.dt:
                    continue
                last = now
                yield Frame(image=img, source=self.label, ts=now)
            cap.release()

    def frames(self) -> Iterator[Frame]:
        return self._stream() if self.mode == "stream" else self._snapshot()
