"""Outputs that need no external service: console, JSONL file, SQLite, image dump."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

import cv2

from thai_alpr.core.plugin import OutputPlugin, register
from thai_alpr.core.types import PlateResult


@register("output", "console")
class ConsoleOutput(OutputPlugin):
    def emit(self, r: PlateResult):
        t = time.strftime("%H:%M:%S", time.localtime(r.ts))
        print(f"[{t}] {r.source:<18} {r.plate or '?':<10} {r.province or '-':<16} {r.plate_type:<10} conf={r.conf:.2f} track={r.track_id}")


@register("output", "jsonl")
class JsonlOutput(OutputPlugin):
    """config: path"""

    def start(self):
        self.path = Path(self.config.get("path", "logs/events.jsonl"))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = self.path.open("a", encoding="utf-8")

    def emit(self, r: PlateResult):
        self.f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
        self.f.flush()

    def stop(self):
        self.f.close()


@register("output", "sqlite")
class SqliteOutput(OutputPlugin):
    """config: path (default data/events.db)"""

    SCHEMA = """CREATE TABLE IF NOT EXISTS events(
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, source TEXT, plate TEXT, province TEXT,
        plate_type TEXT, conf REAL, track_id INTEGER, bbox TEXT, image TEXT)"""

    def start(self):
        path = Path(self.config.get("path", "data/events.db"))
        path.mkdir(parents=True, exist_ok=True) if path.suffix == "" else path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.execute(self.SCHEMA)
        self.db.commit()
        self.lock = threading.Lock()

    def emit(self, r: PlateResult):
        with self.lock:
            self.db.execute(
                "INSERT INTO events(ts,source,plate,province,plate_type,conf,track_id,bbox,image) VALUES(?,?,?,?,?,?,?,?,?)",
                (r.ts, r.source, r.plate, r.province, r.plate_type, r.conf, r.track_id, json.dumps(list(r.bbox)), None),
            )
            self.db.commit()

    def stop(self):
        self.db.close()


@register("output", "save_crops")
class SaveCropsOutput(OutputPlugin):
    """Dump plate crops to disk - handy for building a training set. config: dir"""

    def start(self):
        self.dir = Path(self.config.get("dir", "data/crops"))
        self.dir.mkdir(parents=True, exist_ok=True)

    def emit(self, r: PlateResult):
        if r.plate_crop is None or r.plate_crop.size == 0:
            return
        name = f"{int(r.ts * 1000)}_{(r.plate or 'unk').replace(' ', '')}.jpg"
        cv2.imwrite(str(self.dir / name), r.plate_crop)
