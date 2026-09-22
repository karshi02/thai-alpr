import numpy as np

from thai_alpr.core import plugin as P
from thai_alpr.engine.ocr import PlateReader
from thai_alpr.engine.postprocess import TrackVoter, plate_type_from_color
from thai_alpr.engine.provinces import PROVINCES, match_province


def test_provinces_77():
    assert len(set(PROVINCES)) == 77


def test_fuzzy_province():
    assert match_province("กรุงเทพมหานคร")[0] == "กรุงเทพมหานคร"
    assert match_province("มหาสารคม")[0] == "มหาสารคาม"     # one char dropped
    assert match_province("xyz")[0] == ""


def test_normalise():
    assert PlateReader.normalise("1กข1234") == "1กข 1234"
    assert PlateReader.normalise("กข 12") == "กข 12"
    assert PlateReader.normalise("abc") == ""


def test_plate_type_white():
    img = np.full((60, 200, 3), 245, np.uint8)
    assert plate_type_from_color(img) == "private"


def test_voter():
    v = TrackVoter(window=5, min_votes=3)
    for p in ["1กข 1234", "1กข 1234", "1กข 1284"]:
        plate, prov, conf, new = v.vote(7, p, "ขอนแก่น", 0.9)
    assert plate == "1กข 1234" and new is True
    plate, _, _, new = v.vote(7, "1กข 1234", "ขอนแก่น", 0.9)
    assert new is False


def test_plugins_register():
    P.discover()
    a = P.available()
    assert {"camera", "esp32cam", "http_push", "folder"} <= set(a["input"])
    assert {"mqtt", "webhook", "sqlite", "console"} <= set(a["output"])
    assert {"allowlist", "dedupe", "min_conf"} <= set(a["filter"])
