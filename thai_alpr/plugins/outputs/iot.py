"""Outputs that talk to other systems: MQTT (ESP32 gate/relay), HTTP webhook, serial, LINE."""
from __future__ import annotations

import json

import requests

from thai_alpr.core.plugin import OutputPlugin, register
from thai_alpr.core.types import PlateResult


@register("output", "mqtt")
class MqttOutput(OutputPlugin):
    """Publish each event as JSON. ESP32 subscribes and opens a gate / lights an LED.
    config: host, port (1883), topic (alpr/events), username, password, qos (0), retain (false)
    Extra: `command_topic` + `allow_field` - when the result carries meta allowed=True
    (set by the allowlist filter) we also publish "OPEN" to command_topic."""

    def start(self):
        import paho.mqtt.client as mqtt
        self.topic = self.config.get("topic", "alpr/events")
        self.cmd_topic = self.config.get("command_topic")
        self.qos = int(self.config.get("qos", 0))
        self.client = mqtt.Client(client_id=self.config.get("client_id", "thai-alpr"))
        if self.config.get("username"):
            self.client.username_pw_set(self.config["username"], self.config.get("password"))
        self.client.connect(self.config.get("host", "localhost"), int(self.config.get("port", 1883)), keepalive=30)
        self.client.loop_start()

    def emit(self, r: PlateResult):
        payload = json.dumps(r.to_dict(), ensure_ascii=False)
        self.client.publish(self.topic, payload, qos=self.qos, retain=bool(self.config.get("retain", False)))
        if self.cmd_topic and getattr(r, "allowed", False):
            self.client.publish(self.cmd_topic, "OPEN", qos=1)

    def stop(self):
        self.client.loop_stop()
        self.client.disconnect()


@register("output", "webhook")
class WebhookOutput(OutputPlugin):
    """POST JSON to any URL (Node-RED, Home Assistant, Google Sheets script, your own backend).
    config: url, headers (dict), timeout (3)"""

    def start(self):
        self.url = self.config["url"]
        self.headers = {"Content-Type": "application/json", **self.config.get("headers", {})}
        self.timeout = float(self.config.get("timeout", 3))

    def emit(self, r: PlateResult):
        body = r.to_dict()
        body["allowed"] = bool(getattr(r, "allowed", False))
        requests.post(self.url, json=body, headers=self.headers, timeout=self.timeout)


@register("output", "serial")
class SerialOutput(OutputPlugin):
    """Write one line per event to a serial port (ESP32/Arduino on USB). config: port, baud (115200)
    Line format: PLATE|PROVINCE|TYPE|CONF|ALLOWED\\n"""

    def start(self):
        import serial  # pyserial
        self.ser = serial.Serial(self.config["port"], int(self.config.get("baud", 115200)), timeout=1)

    def emit(self, r: PlateResult):
        line = f"{r.plate}|{r.province}|{r.plate_type}|{r.conf:.2f}|{int(bool(getattr(r, 'allowed', False)))}\n"
        self.ser.write(line.encode("utf-8"))

    def stop(self):
        self.ser.close()


@register("output", "line_notify")
class LineNotifyOutput(OutputPlugin):
    """Push a message (and the crop) to a LINE group. config: token, only_allowed (false)"""

    def start(self):
        self.token = self.config["token"]
        self.only_allowed = bool(self.config.get("only_allowed", False))

    def emit(self, r: PlateResult):
        if self.only_allowed and not getattr(r, "allowed", False):
            return
        import cv2
        msg = f"\n{r.plate} {r.province}\n{r.plate_type} conf={r.conf:.2f}\nfrom {r.source}"
        files = None
        if r.plate_crop is not None and r.plate_crop.size:
            ok, buf = cv2.imencode(".jpg", r.plate_crop)
            if ok:
                files = {"imageFile": ("plate.jpg", buf.tobytes(), "image/jpeg")}
        requests.post("https://notify-api.line.me/api/notify", headers={"Authorization": f"Bearer {self.token}"},
                      data={"message": msg}, files=files, timeout=5)
