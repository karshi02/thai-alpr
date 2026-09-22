"""Download Thai plate datasets from Roboflow Universe in YOLOv8 format.

    set ROBOFLOW_API_KEY=xxxx   (PowerShell: $env:ROBOFLOW_API_KEY="xxxx")
    python scripts/download_data.py plates      # -> data/plates
    python scripts/download_data.py chars       # -> data/chars
    python scripts/download_data.py all

Get a free key: https://app.roboflow.com -> Settings -> API keys.
Dataset pages (check licence before commercial use):
  plates : https://universe.roboflow.com/thailland-plates/thailand-license-plates
           https://universe.roboflow.com/naruesorn/thai-license-plate-j6y9l
  chars  : https://universe.roboflow.com/card-detector/thai-license-plate-character-detect
  prov   : https://universe.roboflow.com/th-support-cytron-yvkeg/province-on-thai-license-plate-detector
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# card-detector labels: A01..A44 = Thai consonants ก..ฮ in dictionary order, A45..A54 = digits 0..9
_CONS = "กขฃคฅฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮ"
CHAR_MAP = {f"A{i:02d}": (_CONS[i - 1] if i <= 44 else str(i - 45)) for i in range(1, 55)}

DATASETS = {
    # name: (workspace, project, version)  - version numbers may move; bump if download 404s
    "plates": ("naruesorn", "thai-license-plate-j6y9l", 1),                       # 349 imgs, class "plate"
    "chars": ("card-detector", "thai-license-plate-character-detect", 1),          # 2521 imgs, classes A01-A54 (see CHAR_MAP)
    "digits": ("dataset-format-conversion-iidaz", "thailand-license-plate-recognition", 1),  # 343 imgs, 0-9
    "province": ("th-support-cytron-yvkeg", "province-on-thai-license-plate-detector", 6),   # 211 imgs, 60 provinces (english)
}


def download(name: str, key: str) -> Path:
    from roboflow import Roboflow
    ws, proj, ver = DATASETS[name]
    out = Path("data") / name
    if (out / "data.yaml").exists():
        print(f"[skip] {out} exists")
        return out
    print(f"[get ] {ws}/{proj} v{ver} -> {out}")
    rf = Roboflow(api_key=key)
    ds = rf.workspace(ws).project(proj).version(ver).download("yolov8", location=str(out))
    fix_yaml(out)
    return Path(ds.location)


def fix_yaml(root: Path) -> None:
    """Roboflow writes relative paths that break when cwd != dataset dir; make them absolute."""
    y = root / "data.yaml"
    if not y.exists():
        return
    import yaml
    d = yaml.safe_load(y.read_text(encoding="utf-8"))
    for key, folder in (("train", "train"), ("val", "valid"), ("test", "test")):
        if key in d and (root / folder / "images").exists():
            d[key] = (root / folder / "images").resolve().as_posix()
    if root.name == "chars":                       # decode A01..A54 -> real characters
        d["names"] = [CHAR_MAP.get(n, n) for n in d["names"]]
    y.write_text(yaml.safe_dump(d, allow_unicode=True, sort_keys=False), encoding="utf-8")


if __name__ == "__main__":
    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        sys.exit("set ROBOFLOW_API_KEY first")
    names = sys.argv[1:] or ["plates", "chars"]
    if names == ["all"]:
        names = list(DATASETS)
    for n in names:
        download(n, key)
