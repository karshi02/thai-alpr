"""Draw boxes + Thai text on frames (OpenCV cannot render Thai, so use PIL)."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from thai_alpr.core.types import PlateResult

_FONT_CANDIDATES = [
    "C:/Windows/Fonts/tahoma.ttf", "C:/Windows/Fonts/LeelawUI.ttf",
    "/usr/share/fonts/truetype/tlwg/Sawasdee.ttf", "/usr/share/fonts/truetype/noto/NotoSansThai-Regular.ttf",
]
_font_cache: dict[int, ImageFont.FreeTypeFont] = {}


def _font(size: int):
    if size not in _font_cache:
        for p in _FONT_CANDIDATES:
            if Path(p).exists():
                _font_cache[size] = ImageFont.truetype(p, size)
                break
        else:
            _font_cache[size] = ImageFont.load_default()
    return _font_cache[size]


def annotate(image: np.ndarray, results: list[PlateResult]) -> np.ndarray:
    out = image.copy()
    for r in results:
        x1, y1, x2, y2 = r.bbox
        color = (0, 200, 0) if getattr(r, "allowed", False) else (0, 140, 255) if r.plate else (0, 0, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
    pil = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    size = max(16, image.shape[0] // 30)
    for r in results:
        x1, y1, _, _ = r.bbox
        label = f"{r.plate or '?'}  {r.province or ''}  {r.conf:.2f}"
        w = d.textlength(label, font=_font(size))
        top = y1 - size - 6 if y1 - size - 6 >= 0 else y1 + 2      # drop inside the box when at the frame edge
        d.rectangle([x1, top, x1 + w + 8, top + size + 6], fill=(0, 0, 0))
        d.text((x1 + 4, top + 2), label, font=_font(size), fill=(255, 255, 255))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
