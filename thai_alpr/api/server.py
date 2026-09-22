"""FastAPI server: REST + WebSocket + dashboard.

  POST /detect                 multipart image -> plates JSON (+ ?annotate=1 returns JPEG)
  POST /ingest/{device}        ESP32-CAM pushes a JPEG into the running pipeline
  GET  /events?limit=50        recent events from the SQLite output
  GET  /stats                  fps, counters, plugins
  GET  /plugins                registered plugin types
  WS   /ws                     live events as JSON
  GET  /                       dashboard
"""
from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, Response

from thai_alpr import __version__
from thai_alpr.core import plugin as P
from thai_alpr.core.pipeline import Pipeline, load_config
from thai_alpr.core.types import Frame

from .draw import annotate

STATIC = Path(__file__).parent / "static"


def create_app(config_path: str = "configs/default.yaml") -> FastAPI:
    cfg = load_config(config_path)
    # make sure an http_push input exists so /ingest works even if the yaml lacks one
    if not any(i.get("type") == "http_push" for i in cfg.get("inputs", [])):
        cfg.setdefault("inputs", []).append({"type": "http_push", "id": "api"})
    pipe = Pipeline(cfg)
    app = FastAPI(title="Thai ALPR", version=__version__)
    sockets: set[WebSocket] = set()
    loop_holder: dict = {}

    def broadcast(result):
        loop = loop_holder.get("loop")
        if not loop or not sockets:
            return
        msg = json.dumps(result.to_dict(), ensure_ascii=False)
        for ws in list(sockets):
            asyncio.run_coroutine_threadsafe(ws.send_text(msg), loop)

    pipe.listeners.append(broadcast)

    @app.on_event("startup")
    async def _start():
        loop_holder["loop"] = asyncio.get_running_loop()
        pipe.start()

    @app.on_event("shutdown")
    async def _stop():
        pipe.stop()

    @app.get("/", response_class=HTMLResponse)
    def index():
        return (STATIC / "index.html").read_text(encoding="utf-8")

    @app.post("/detect")
    async def detect(file: UploadFile = File(...), annotate_img: int = Query(0, alias="annotate")):
        data = await file.read()
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return JSONResponse({"error": "bad image"}, status_code=400)
        t = time.time()
        results = pipe.process_frame(Frame(image=img, source="api", meta={"still": True}))
        if annotate_img:
            ok, buf = cv2.imencode(".jpg", annotate(img, results))
            return Response(buf.tobytes(), media_type="image/jpeg")
        return {"ms": round((time.time() - t) * 1000), "plates": [r.to_dict() for r in results]}

    @app.post("/ingest/{device}")
    async def ingest(device: str, file: UploadFile = File(...)):
        from thai_alpr.plugins.inputs.http_push import HttpPushInput
        inp = HttpPushInput.any_instance()
        if inp is None:
            return JSONResponse({"error": "no http_push input"}, status_code=503)
        ok = inp.push(await file.read(), device)
        return {"queued": ok}

    @app.get("/events")
    def events(limit: int = 50):
        db_cfg = next((o for o in cfg.get("outputs", []) if o.get("type") == "sqlite" and o.get("enabled", True)), None)
        if not db_cfg:
            return []
        con = sqlite3.connect(db_cfg.get("path", "data/events.db"))
        con.row_factory = sqlite3.Row
        rows = con.execute("SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        con.close()
        return [dict(r) for r in rows]

    @app.get("/stats")
    def stats():
        return {
            "version": __version__, **pipe.stats,
            "inputs": [i.label for i in pipe.inputs], "filters": [f.label for f in pipe.filters], "outputs": [o.label for o in pipe.outputs],
        }

    @app.get("/plugins")
    def plugins():
        return P.available()

    @app.websocket("/ws")
    async def ws(ws: WebSocket):
        await ws.accept()
        sockets.add(ws)
        try:
            while True:
                await ws.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            sockets.discard(ws)

    return app
