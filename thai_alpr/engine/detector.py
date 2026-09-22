"""Stage 1: locate plates in a frame with YOLOv8 and return corrected crops.

Until a Thai-trained weight exists in models/plate.pt we fall back to the
generic COCO yolov8n.pt (which does NOT know "license plate") so the pipeline
still runs; train with scripts/train_plate.py to get real detections.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class PlateDetector:
    def __init__(self, weights: str = "models/plate.pt", conf: float = 0.4, imgsz: int = 640, device: str | None = None):
        from ultralytics import YOLO

        path = Path(weights)
        self.untrained = not path.exists()
        if self.untrained:
            print(f"[detector] {weights} not found - treating the whole frame as one plate (train with scripts/train.py plate)")
        self.model = None if self.untrained else YOLO(weights)
        self.conf = conf
        self.imgsz = imgsz
        self.device = device

    def detect(self, image: np.ndarray, track: bool = False) -> list[dict]:
        """Return [{bbox:(x1,y1,x2,y2), conf, track_id}] sorted by confidence."""
        if self.untrained:
            h, w = image.shape[:2]
            return [{"bbox": (0, 0, w, h), "conf": 1.0, "track_id": None}]
        kwargs = dict(conf=self.conf, imgsz=self.imgsz, verbose=False)
        if self.device:
            kwargs["device"] = self.device
        if track:
            res = self.model.track(image, persist=True, tracker="bytetrack.yaml", **kwargs)[0]
        else:
            res = self.model.predict(image, **kwargs)[0]

        out = []
        if res.boxes is None:
            return out
        ids = res.boxes.id.int().tolist() if res.boxes.id is not None else [None] * len(res.boxes)
        for box, c, tid in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist(), ids):
            x1, y1, x2, y2 = (int(v) for v in box)
            out.append({"bbox": (x1, y1, x2, y2), "conf": float(c), "track_id": tid})
        return sorted(out, key=lambda d: -d["conf"])

    @staticmethod
    def crop(image: np.ndarray, bbox: tuple[int, int, int, int], pad: float = 0.05) -> np.ndarray:
        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox
        px, py = int((x2 - x1) * pad), int((y2 - y1) * pad)
        x1, y1 = max(0, x1 - px), max(0, y1 - py)
        x2, y2 = min(w, x2 + px), min(h, y2 + py)
        return image[y1:y2, x1:x2]

    @staticmethod
    def deskew(crop: np.ndarray) -> np.ndarray:
        """Best-effort perspective correction: find the largest 4-point contour and warp it.
        Falls back to the untouched crop when no clean quadrilateral is found."""
        if crop.size == 0:
            return crop
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        edges = cv2.Canny(gray, 50, 150)
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        area_min = 0.4 * crop.shape[0] * crop.shape[1]
        for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:5]:
            if cv2.contourArea(c) < area_min:
                break
            approx = cv2.approxPolyDP(c, 0.03 * cv2.arcLength(c, True), True)
            if len(approx) == 4:
                pts = approx.reshape(4, 2).astype(np.float32)
                s, d = pts.sum(1), np.diff(pts, axis=1).ravel()
                src = np.array([pts[s.argmin()], pts[d.argmin()], pts[s.argmax()], pts[d.argmax()]], np.float32)
                w = int(max(np.linalg.norm(src[1] - src[0]), np.linalg.norm(src[2] - src[3])))
                h = int(max(np.linalg.norm(src[3] - src[0]), np.linalg.norm(src[2] - src[1])))
                if w < 20 or h < 10:
                    break
                dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
                M = cv2.getPerspectiveTransform(src, dst)
                return cv2.warpPerspective(crop, M, (w, h))
        return crop
