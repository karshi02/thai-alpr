"""Orchestrates: input plugins -> engine (3 stages) -> filters -> output plugins.

Every input runs in its own thread and pushes Frames into one queue; a single
worker thread owns the GPU and processes frames in order. Outputs are called
synchronously from the worker (keep them fast; use the webhook/mqtt plugins
for anything slow).
"""
from __future__ import annotations

import queue
import threading
import time
from typing import Any

import yaml

from . import plugin as P
from .types import Frame, PlateResult


class Pipeline:
    def __init__(self, config: dict[str, Any]):
        self.cfg = config
        P.discover()
        self.inputs: list[P.InputPlugin] = [P.create("input", c) for c in config.get("inputs", []) if c.get("enabled", True)]
        self.filters: list[P.FilterPlugin] = [P.create("filter", c) for c in config.get("filters", []) if c.get("enabled", True)]
        self.outputs: list[P.OutputPlugin] = [P.create("output", c) for c in config.get("outputs", []) if c.get("enabled", True)]

        eng = config.get("engine", {})
        from thai_alpr.engine.detector import PlateDetector
        from thai_alpr.engine.ocr import PlateReader
        from thai_alpr.engine.postprocess import TrackVoter, plate_type_from_color

        self.detector = PlateDetector(**eng.get("detector", {}))
        self.reader = PlateReader(**eng.get("reader", {}))
        self.voter = TrackVoter(**eng.get("voter", {}))
        self.plate_type = plate_type_from_color
        self.track = bool(eng.get("track", True))
        self.emit_every_frame = bool(eng.get("emit_every_frame", False))

        self._q: queue.Queue[Frame] = queue.Queue(maxsize=int(config.get("queue_size", 8)))
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self.stats = {"frames": 0, "plates": 0, "emitted": 0, "dropped": 0, "fps": 0.0}
        self.last_results: list[PlateResult] = []
        self.listeners: list = []          # callables(result) e.g. API websocket broadcaster

    # ---- single image (used by API /detect and CLI) -------------------------------
    def process_frame(self, frame: Frame) -> list[PlateResult]:
        results: list[PlateResult] = []
        dets = self.detector.detect(frame.image, track=self.track and not frame.meta.get("still", False))
        for d in dets:
            crop = self.detector.crop(frame.image, d["bbox"])
            crop = self.detector.deskew(crop)
            r = self.reader.read(crop)
            conf = d["conf"] * (r["plate_conf"] if r["plate"] else 0.0)
            results.append(PlateResult(
                plate=r["plate"], province=r["province"], plate_type=self.plate_type(crop),
                conf=conf, bbox=d["bbox"], source=frame.source, ts=frame.ts,
                track_id=d["track_id"], raw_chars=r["chars"], plate_crop=crop,
            ))
        self.stats["frames"] += 1
        self.stats["plates"] += len(results)
        self.last_results = results
        return results

    def _dispatch(self, res: PlateResult) -> None:
        for f in self.filters:
            res = f.apply(res)
            if res is None:
                return
        for o in self.outputs:
            try:
                o.emit(res)
            except Exception as e:      # one bad output must not kill the loop
                print(f"[pipeline] output {o.label} failed: {e}")
        for cb in self.listeners:
            try:
                cb(res)
            except Exception:
                pass
        self.stats["emitted"] += 1

    # ---- streaming ----------------------------------------------------------------
    def _pump(self, inp: P.InputPlugin) -> None:
        try:
            inp.start()
            for fr in inp.frames():
                if self._stop.is_set():
                    break
                try:
                    self._q.put(fr, timeout=0.5)
                except queue.Full:
                    self.stats["dropped"] += 1
        except Exception as e:
            print(f"[pipeline] input {inp.label} stopped: {e}")
        finally:
            inp.stop()

    def _worker(self) -> None:
        t0, n = time.time(), 0
        while not self._stop.is_set():
            try:
                fr = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            for res in self.process_frame(fr):
                if self.track and res.track_id is not None:
                    plate, prov, conf, is_new = self.voter.vote(res.track_id, res.plate, res.province, res.conf)
                    res.plate, res.province, res.conf = plate, prov, conf
                    if not (is_new or self.emit_every_frame):
                        continue
                elif not res.plate:
                    continue
                self._dispatch(res)
            if self.track:
                self.voter.forget({r.track_id for r in self.last_results if r.track_id is not None})
            n += 1
            if time.time() - t0 >= 2:
                self.stats["fps"] = round(n / (time.time() - t0), 1)
                t0, n = time.time(), 0

    def start(self) -> None:
        for o in self.outputs:
            o.start()
        for f in self.filters:
            f.start()
        self._threads = [threading.Thread(target=self._pump, args=(i,), daemon=True, name=i.label) for i in self.inputs]
        self._threads.append(threading.Thread(target=self._worker, daemon=True, name="worker"))
        for t in self._threads:
            t.start()

    def stop(self) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=2)
        for p in self.outputs + self.filters:
            p.stop()

    def run_forever(self) -> None:
        self.start()
        try:
            while any(t.is_alive() for t in self._threads[:-1]) or not self._q.empty():
                time.sleep(0.2)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()


def load_config(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
