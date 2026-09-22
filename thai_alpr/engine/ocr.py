"""Stage 2 + 3: read the top line (chars) and bottom line (province) of a plate crop.

Strategy
  top line   : YOLO char detector (models/chars.pt) -> sort by x -> regex
               fallback: EasyOCR 'th' when chars.pt is missing
  bottom line: EasyOCR 'th' -> fuzzy match against 77 provinces
               (swap for a 77-class classifier in models/province.pt later)
"""
from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np

from .provinces import en_to_th, match_province

# 1กข 1234 | กข 1234 | 1กข 123 ... digits optional prefix, 1-2 thai consonants, 1-4 digits
PLATE_RE = re.compile(r"^(\d?)([ก-ฮ]{1,2})\s?(\d{1,4})$")
THAI_CONSONANTS = "กขฃคฅฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮ"
ALLOWED = set("0123456789" + THAI_CONSONANTS)


class PlateReader:
    def __init__(self, chars_weights: str = "models/chars.pt", province_weights: str = "models/province.pt",
                 conf: float = 0.3, top_ratio: float = 0.62, use_gpu: bool = True):
        self.top_ratio = top_ratio
        self.conf = conf
        self.char_model = None
        self.province_model = None
        self._easy = None
        self._use_gpu = use_gpu
        if Path(province_weights).exists():
            from ultralytics import YOLO
            self.province_model = YOLO(province_weights)
        else:
            print(f"[ocr] {province_weights} not found, province will use EasyOCR + fuzzy match")
        if Path(chars_weights).exists():
            from ultralytics import YOLO
            self.char_model = YOLO(chars_weights)
        else:
            print(f"[ocr] {chars_weights} not found, top line will use EasyOCR fallback")

    # ---- helpers -----------------------------------------------------------------
    @property
    def easy(self):
        if self._easy is None:
            import easyocr
            self._easy = easyocr.Reader(["th", "en"], gpu=self._use_gpu, verbose=False)
        return self._easy

    @staticmethod
    def _enhance(img: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return cv2.cvtColor(clahe.apply(gray), cv2.COLOR_GRAY2BGR)

    def split(self, crop: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = crop.shape[0]
        cut = int(h * self.top_ratio)
        return crop[:cut], crop[cut:]

    # ---- stage 2 -----------------------------------------------------------------
    def read_top(self, top: np.ndarray) -> tuple[str, float, list[dict]]:
        if top.size == 0:
            return "", 0.0, []
        top = cv2.resize(top, None, fx=max(1.0, 320 / max(1, top.shape[1])), fy=max(1.0, 320 / max(1, top.shape[1])))
        top = self._enhance(top)

        chars: list[dict] = []
        if self.char_model is not None:
            res = self.char_model.predict(top, conf=self.conf, imgsz=320, verbose=False)[0]
            names = res.names
            for box, c, k in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist(), res.boxes.cls.tolist()):
                chars.append({"char": names[int(k)], "conf": float(c), "x": (box[0] + box[2]) / 2})
        else:
            for bbox, text, c in self.easy.readtext(top, allowlist="".join(sorted(ALLOWED)) + " "):
                x = sum(p[0] for p in bbox) / 4
                for i, ch in enumerate(text.replace(" ", "")):
                    chars.append({"char": ch, "conf": float(c), "x": x + i})

        chars.sort(key=lambda d: d["x"])
        text = "".join(d["char"] for d in chars if d["char"] in ALLOWED)
        conf = float(np.mean([d["conf"] for d in chars])) if chars else 0.0
        return self.normalise(text), conf, chars

    @staticmethod
    def normalise(text: str) -> str:
        """Insert the space between letters and number, validate with regex."""
        text = text.replace(" ", "")
        m = re.match(r"^(\d?)([ก-ฮ]{1,2})(\d{1,4})$", text)
        if not m:
            return ""
        prefix, letters, number = m.groups()
        return f"{prefix}{letters} {number}"

    # ---- stage 3 -----------------------------------------------------------------
    def read_province(self, bottom: np.ndarray, whole: np.ndarray | None = None) -> tuple[str, float]:
        if bottom.size == 0 or bottom.shape[0] < 8:
            return "", 0.0
        if self.province_model is not None:
            # detector trained on whole plates: run on the full crop, take the best province box
            res = self.province_model.predict(whole if whole is not None else bottom, conf=0.2, imgsz=320, verbose=False)[0]
            if len(res.boxes):
                i = int(res.boxes.conf.argmax())
                return en_to_th(res.names[int(res.boxes.cls[i])]), float(res.boxes.conf[i])
            return "", 0.0
        bottom = cv2.resize(bottom, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        bottom = self._enhance(bottom)
        hits = self.easy.readtext(bottom, detail=1, paragraph=False)
        if not hits:
            return "", 0.0
        text = "".join(h[1] for h in hits)
        prov, score = match_province(text)
        ocr_conf = float(np.mean([h[2] for h in hits]))
        return prov, score * ocr_conf if prov else 0.0

    # ---- combined ----------------------------------------------------------------
    def read(self, crop: np.ndarray) -> dict:
        top, bottom = self.split(crop)
        plate, pconf, chars = self.read_top(top)
        province, vconf = self.read_province(bottom, whole=crop)
        return {"plate": plate, "plate_conf": pconf, "province": province, "province_conf": vconf, "chars": chars}
