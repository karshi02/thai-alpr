"""Plugin base classes + registry.

Three plugin kinds:
  input   - yields Frames (webcam, RTSP, ESP32-CAM, folder, HTTP upload ...)
  filter  - optional hook run on each PlateResult before outputs (dedupe, allowlist ...)
  output  - consumes PlateResults (console, SQLite, MQTT -> ESP32 gate, webhook, LINE ...)

Register with the decorator:

    @register("output", "mqtt")
    class MqttOutput(OutputPlugin): ...

Plugins are discovered by importing every module under thai_alpr/plugins/.
Third-party packages can also register via the entry-point group "thai_alpr.plugins".
"""
from __future__ import annotations

import importlib
import pkgutil
from typing import Any, Callable, Iterator

from .types import Frame, PlateResult

KINDS = ("input", "filter", "output")
_REGISTRY: dict[str, dict[str, type]] = {k: {} for k in KINDS}


class Plugin:
    """Common lifecycle. `config` is the dict from configs/*.yaml for this instance."""
    kind: str = ""
    name: str = ""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.instance = self.config.get("id", self.name)

    def start(self) -> None:      # open connections / devices
        pass

    def stop(self) -> None:       # release resources
        pass

    @property
    def label(self) -> str:
        return f"{self.name}:{self.instance}"


class InputPlugin(Plugin):
    kind = "input"

    def frames(self) -> Iterator[Frame]:
        """Blocking generator of Frames. Return when the source ends."""
        raise NotImplementedError


class FilterPlugin(Plugin):
    kind = "filter"

    def apply(self, result: PlateResult) -> PlateResult | None:
        """Return the (possibly modified) result, or None to drop it."""
        return result


class OutputPlugin(Plugin):
    kind = "output"

    def emit(self, result: PlateResult) -> None:
        raise NotImplementedError


def register(kind: str, name: str) -> Callable[[type], type]:
    if kind not in KINDS:
        raise ValueError(f"unknown plugin kind {kind!r}, expected one of {KINDS}")

    def deco(cls: type) -> type:
        cls.kind = kind
        cls.name = name
        _REGISTRY[kind][name] = cls
        return cls
    return deco


def discover() -> None:
    """Import every module in thai_alpr.plugins.* so their @register runs."""
    import thai_alpr.plugins as pkg
    for mod in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
        importlib.import_module(mod.name)

    # external packages
    try:
        from importlib.metadata import entry_points
        for ep in entry_points(group="thai_alpr.plugins"):
            ep.load()
    except Exception:
        pass


def get(kind: str, name: str) -> type:
    try:
        return _REGISTRY[kind][name]
    except KeyError:
        available = ", ".join(sorted(_REGISTRY.get(kind, {}))) or "(none)"
        raise KeyError(f"no {kind} plugin named {name!r}; available: {available}")


def create(kind: str, config: dict[str, Any]) -> Plugin:
    """Instantiate from a config entry like {"type": "mqtt", "host": ...}."""
    cls = get(kind, config["type"])
    return cls(config)


def available() -> dict[str, list[str]]:
    return {k: sorted(v) for k, v in _REGISTRY.items()}
