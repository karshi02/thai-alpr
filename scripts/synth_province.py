"""Generate synthetic bottom-line crops for all 77 provinces.

Real province crops are scarce (Roboflow set has 211 images). Rendering the
province names with a Thai font + heavy augmentation gives thousands of samples
to train a 77-class classifier (models/province.pt) later.

    python scripts/synth_province.py --per 60      # -> data/province_synth/<province>/*.jpg
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from thai_alpr.engine.provinces import PROVINCES  # noqa: E402

FONTS = [p for p in [
    "C:/Windows/Fonts/tahomabd.ttf", "C:/Windows/Fonts/tahoma.ttf", "C:/Windows/Fonts/LeelaUIb.ttf",
    "C:/Windows/Fonts/LeelawUI.ttf", "C:/Windows/Fonts/cordiab.ttf", "C:/Windows/Fonts/browab.ttf",
] if Path(p).exists()]
if not FONTS:
    sys.exit("no Thai font found - add a .ttf path to FONTS")

BACKGROUNDS = [(255, 255, 255), (250, 250, 240), (255, 210, 40), (90, 170, 90), (220, 60, 60)]  # white, cream, yellow, green, red


def render(text: str) -> np.ndarray:
    font = ImageFont.truetype(random.choice(FONTS), random.randint(34, 46))
    bg = random.choice(BACKGROUNDS)
    fg = (0, 0, 0) if sum(bg) > 400 else (255, 255, 255)
    w = int(font.getlength(text)) + random.randint(20, 80)
    h = random.randint(56, 72)
    img = Image.new("RGB", (w, h), bg)
    d = ImageDraw.Draw(img)
    d.text(((w - font.getlength(text)) / 2, (h - 46) / 2 + random.randint(-3, 3)), text, font=font, fill=fg)
    a = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # augment: perspective, blur, noise, brightness, jpeg
    H, W = a.shape[:2]
    j = 0.06
    src = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
    dst = src + np.random.uniform(-j, j, (4, 2)) * [W, H]
    a = cv2.warpPerspective(a, cv2.getPerspectiveTransform(src, dst.astype(np.float32)), (W, H), borderValue=bg[::-1])
    if random.random() < 0.6:
        a = cv2.GaussianBlur(a, (random.choice([3, 5]),) * 2, 0)
    a = np.clip(a.astype(np.int16) + np.random.normal(0, random.uniform(2, 14), a.shape), 0, 255).astype(np.uint8)
    a = cv2.convertScaleAbs(a, alpha=random.uniform(0.6, 1.3), beta=random.randint(-40, 30))
    a = cv2.resize(a, (random.randint(90, 200), random.randint(22, 44)))  # down to realistic crop size
    _, enc = cv2.imencode(".jpg", a, [cv2.IMWRITE_JPEG_QUALITY, random.randint(35, 85)])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--per", type=int, default=60)
    ap.add_argument("--out", default="data/province_synth")
    a = ap.parse_args()
    for i, prov in enumerate(PROVINCES):
        d = Path(a.out) / f"{i:02d}_{prov}"
        d.mkdir(parents=True, exist_ok=True)
        for k in range(a.per):
            cv2.imencode(".jpg", render(prov))[1].tofile(str(d / f"{k:04d}.jpg"))
    print(f"wrote {77 * a.per} images to {a.out}")
