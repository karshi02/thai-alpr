"""CLI.

  python -m thai_alpr.cli run   -c configs/default.yaml      # stream from configured inputs
  python -m thai_alpr.cli image path/to/photo.jpg           # one image, print JSON, save annotated
  python -m thai_alpr.cli plugins                            # list registered plugins
  python -m thai_alpr.cli serve -c configs/default.yaml      # FastAPI + dashboard on :8000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from thai_alpr.core import plugin as P
from thai_alpr.core.pipeline import Pipeline, load_config
from thai_alpr.core.types import Frame


def cmd_plugins(_):
    P.discover()
    for kind, names in P.available().items():
        print(f"{kind:>7}: {', '.join(names)}")


def cmd_image(args):
    import cv2
    cfg = load_config(args.config)
    cfg["inputs"] = []
    cfg["engine"]["track"] = False
    pipe = Pipeline(cfg)
    for p in args.paths:
        img = cv2.imread(p)
        if img is None:
            print(f"cannot read {p}", file=sys.stderr)
            continue
        results = pipe.process_frame(Frame(image=img, source="cli", meta={"still": True}))
        print(json.dumps([r.to_dict() for r in results], ensure_ascii=False, indent=2))
        if args.save:
            from thai_alpr.api.draw import annotate
            out = Path(args.save) / (Path(p).stem + "_out.jpg")
            out.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out), annotate(img, results))
            print(f"saved {out}")


def cmd_run(args):
    pipe = Pipeline(load_config(args.config))
    print(f"inputs={[i.label for i in pipe.inputs]} outputs={[o.label for o in pipe.outputs]}")
    pipe.run_forever()


def cmd_serve(args):
    import uvicorn
    from thai_alpr.api.server import create_app
    app = create_app(args.config)
    uvicorn.run(app, host=args.host, port=args.port)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="thai_alpr")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("plugins").set_defaults(fn=cmd_plugins)

    s = sub.add_parser("image")
    s.add_argument("paths", nargs="+")
    s.add_argument("-c", "--config", default="configs/default.yaml")
    s.add_argument("--save", help="folder for annotated images")
    s.set_defaults(fn=cmd_image)

    s = sub.add_parser("run")
    s.add_argument("-c", "--config", default="configs/default.yaml")
    s.set_defaults(fn=cmd_run)

    s = sub.add_parser("serve")
    s.add_argument("-c", "--config", default="configs/default.yaml")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(fn=cmd_serve)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
