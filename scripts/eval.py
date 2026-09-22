"""End-to-end accuracy on a labelled test folder.

Folder layout: data/test_images/<anything>.jpg + labels.csv with columns
    file,plate,province
e.g.  car1.jpg,1กข 1234,กรุงเทพมหานคร

    python scripts/eval.py data/test_images
Prints plate string accuracy, province accuracy, per-char accuracy and mean latency.
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from thai_alpr.core.pipeline import Pipeline, load_config  # noqa: E402
from thai_alpr.core.types import Frame  # noqa: E402


def main(folder: str, config: str = "configs/default.yaml"):
    root = Path(folder)
    rows = list(csv.DictReader((root / "labels.csv").open(encoding="utf-8")))
    cfg = load_config(config)
    cfg["inputs"], cfg["engine"]["track"] = [], False
    pipe = Pipeline(cfg)

    n = plate_ok = prov_ok = det_ok = 0
    chars_total = chars_ok = 0
    lat = []
    for r in rows:
        img = cv2.imread(str(root / r["file"]))
        if img is None:
            continue
        n += 1
        t = time.time()
        res = pipe.process_frame(Frame(image=img, source="eval", meta={"still": True}))
        lat.append(time.time() - t)
        if not res:
            print(f"MISS  {r['file']}")
            continue
        det_ok += 1
        best = res[0]
        gt, pr = r["plate"].replace(" ", ""), best.plate.replace(" ", "")
        plate_ok += gt == pr
        prov_ok += r.get("province", "") == best.province
        chars_total += len(gt)
        chars_ok += sum(a == b for a, b in zip(gt, pr))
        flag = "ok  " if gt == pr else "BAD "
        print(f"{flag} {r['file']:<24} gt={r['plate']:<10} pred={best.plate:<10} prov={best.province}")

    if not n:
        return
    print(f"\nimages           {n}")
    print(f"plate detected   {det_ok / n:.1%}")
    print(f"plate exact      {plate_ok / n:.1%}")
    print(f"char accuracy    {chars_ok / max(1, chars_total):.1%}")
    print(f"province exact   {prov_ok / n:.1%}")
    print(f"latency mean     {1000 * sum(lat) / len(lat):.0f} ms")


if __name__ == "__main__":
    main(*sys.argv[1:])
