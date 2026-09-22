"""Train Stage 1 (plate box) or Stage 2 (chars) with settings tuned for a 4 GB GPU.

    python scripts/train.py plate --epochs 80          # data/plates/data.yaml -> models/plate.pt
    python scripts/train.py chars --epochs 120         # data/chars/data.yaml  -> models/chars.pt
    python scripts/train.py province --epochs 100      # data/province/data.yaml -> models/province.pt (77 provinces)
    python scripts/train.py plate --data data/merged/data.yaml --model yolov8s.pt

Then export for faster inference:
    python scripts/train.py export models/plate.pt
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def train(task: str, data: str | None, model: str, epochs: int, imgsz: int, batch: int):
    from ultralytics import YOLO
    data = data or {"plate": "data/plates/data.yaml", "chars": "data/chars/data.yaml", "province": "data/province/data.yaml"}[task]
    if not Path(data).exists():
        raise SystemExit(f"{data} not found - run scripts/download_data.py first")
    y = YOLO(model)
    res = y.train(
        data=data, epochs=epochs, imgsz=imgsz, batch=batch, amp=True, workers=0,          # workers=0: this laptop has 8 GB RAM, each worker costs ~400 MB
        project="runs", name=task, exist_ok=True, patience=20, cos_lr=True,
        # augmentation: plates are photographed at night / in rain / at angles
        hsv_h=0.015, hsv_s=0.6, hsv_v=0.5, degrees=5, translate=0.1, scale=0.4, shear=2,
        perspective=0.0005, fliplr=0.0,        # never mirror text
        mosaic=1.0 if task == "plate" else 0.3, close_mosaic=10,
    )
    best = Path(res.save_dir) / "weights" / "best.pt"
    out = Path("models") / f"{task}.pt"
    out.parent.mkdir(exist_ok=True)
    shutil.copy(best, out)
    print(f"\nsaved {out}\nval metrics: {res.results_dict}")


def export(weights: str):
    from ultralytics import YOLO
    YOLO(weights).export(format="onnx", imgsz=640, half=True, simplify=True, dynamic=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["plate", "chars", "province", "export"])
    ap.add_argument("weights", nargs="?", help="for export")
    ap.add_argument("--data")
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args()
    if a.task == "export":
        export(a.weights or "models/plate.pt")
    else:
        small = a.task in ("chars", "province")          # crops are small -> 320px, bigger batch fits 4 GB
        train(a.task, a.data, a.model, a.epochs, 320 if small else a.imgsz, 32 if small else a.batch)
