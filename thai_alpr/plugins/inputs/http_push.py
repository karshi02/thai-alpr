"""Input fed by the API: devices POST a JPEG to /ingest/<device_id>.

Good for ESP32-CAM boards on battery/deep-sleep that push one photo when a
PIR/ultrasonic sensor fires instead of streaming continuously.
"""
from __future__ import annotations

import queue
import time
from typing import Iterator

import cv2
import numpy as np

from thai_alpr.core.plugin import InputPlugin, register
from thai_alpr.core.types import Frame

_INSTANCES: dict[str, "HttpPushInput"] = {}


@register("input", "http_push")
class HttpPushInput(InputPlugin):
    def start(self):
        self.q: queue.Queue[Frame] = queue.Queue(maxsize=int(self.config.get("queue", 32)))
        _INSTANCES[self.instance] = self

    def push(self, jpeg: bytes, device: str) -> bool:
        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return False
        try:
            self.q.put_nowait(Frame(image=img, source=f"{self.label}/{device}", ts=time.time(), meta={"device": device, "still": True}))
            return True
        except queue.Full:
            return False

    def frames(self) -> Iterator[Frame]:
        while True:
            try:
                yield self.q.get(timeout=1)
            except queue.Empty:
                continue

    @staticmethod
    def any_instance() -> "HttpPushInput | None":
        return next(iter(_INSTANCES.values()), None)
