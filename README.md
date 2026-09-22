# Thai ALPR — ระบบอ่านป้ายทะเบียนรถไทยแบบ plugin

อ่านป้ายทะเบียนไทยจากกล้อง / วิดีโอ / ESP32-CAM แล้วส่งผลไปได้หลายทาง (SQLite, MQTT ไป ESP32 เปิดไม้กั้น, webhook, LINE, serial) ทุกส่วนเป็น plugin เปิด-ปิดจาก YAML ไฟล์เดียว

```
กล้อง / ESP32-CAM / รูป ──► Stage 1 YOLOv8 กรอบป้าย ──► deskew
                                                        ├─► Stage 2 YOLO ตัวอักษร (บรรทัดบน)
                                                        └─► Stage 3 OCR + fuzzy 77 จังหวัด (บรรทัดล่าง)
                             ──► filters (min_conf, dedupe, allowlist) ──► outputs (console, sqlite, mqtt, webhook, serial, LINE)
                             ──► FastAPI dashboard + WebSocket live
```

## เริ่มใช้ 5 นาที

```powershell
# 1. env (Python 3.10/3.11 + CUDA 12.1)
python -m venv .venv; .venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 2. ทดสอบรูปเดียว (ยังไม่มี model ไทย -> ใช้ EasyOCR fallback)
python -m thai_alpr.cli image path\to\car.jpg --save out

# 3. รัน dashboard + API  ->  http://localhost:8000
python -m thai_alpr.cli serve -c configs/default.yaml

# 4. รันแบบ headless จาก webcam/RTSP/ESP32 ตาม config
python -m thai_alpr.cli run -c configs/default.yaml
```

## เทรน model ไทย

```powershell
$env:ROBOFLOW_API_KEY = "xxxx"
python scripts/download_data.py plates chars     # Roboflow Universe -> data/
python scripts/train.py plate --epochs 80        # -> models/plate.pt
python scripts/train.py chars --epochs 120       # -> models/chars.pt
python scripts/synth_province.py --per 60        # synthetic จังหวัด 77 class
python scripts/eval.py data/test_images          # accuracy end-to-end
python scripts/train.py export models/plate.pt   # ONNX
```

## Plugins

| kind | type | ใช้ทำอะไร |
|---|---|---|
| input | `camera` | webcam index / ไฟล์วิดีโอ / RTSP |
| input | `rtsp` | กล้อง IP |
| input | `esp32cam` | ESP32-CAM ดึงภาพ `/capture` หรือ MJPEG stream |
| input | `http_push` | อุปกรณ์ POST รูปมาที่ `/ingest/<device>` (ESP32 + PIR) |
| input | `folder` | รูปในโฟลเดอร์ (ทดสอบ) |
| filter | `min_conf` | ตัดผลที่ conf ต่ำ |
| filter | `dedupe` | กันอ่านซ้ำภายใน N วินาที |
| filter | `allowlist` | ติดธง `allowed` จาก `configs/allowlist.txt` ใช้เปิดประตู |
| output | `console` / `jsonl` / `sqlite` / `save_crops` | log ในเครื่อง |
| output | `mqtt` | ส่ง event + สั่ง `OPEN` ไป ESP32 (`firmware/esp32_gate_mqtt`) |
| output | `webhook` | POST JSON ไป Node-RED / Home Assistant / backend |
| output | `serial` | ESP32/Arduino ทาง USB |
| output | `line_notify` | แจ้งเตือน LINE พร้อมรูปป้าย |

เขียน plugin ใหม่ = ไฟล์เดียวใน `thai_alpr/plugins/<kind>/`:

```python
from thai_alpr.core.plugin import OutputPlugin, register

@register("output", "telegram")
class TelegramOutput(OutputPlugin):
    def emit(self, r):
        ...  # r.plate, r.province, r.plate_type, r.conf, r.allowed, r.plate_crop
```

แล้วเพิ่ม `- type: telegram` ใน YAML จบ

## API

| method | path | |
|---|---|---|
| POST | `/detect` | multipart `file` -> JSON; `?annotate=1` คืน JPEG วาดกรอบ |
| POST | `/ingest/{device}` | ESP32 ส่งรูปเข้า pipeline |
| GET | `/events?limit=50` | เหตุการณ์จาก SQLite |
| GET | `/stats` `/plugins` | สถานะ |
| WS | `/ws` | event สดเป็น JSON |

## Hardware ที่รองรับ

- **ESP32-CAM** — `firmware/esp32cam_streamer/` ทั้ง pull (`/capture`) และ push (`/ingest`) + trigger PIR
- **ESP32 + relay** — `firmware/esp32_gate_mqtt/` รับ MQTT เปิดไม้กั้น/ไฟ/บัซเซอร์
- กล้อง IP RTSP, webcam USB, ไฟล์วิดีโอ

## โครงโปรเจก

```
thai_alpr/
  core/      types.py plugin.py pipeline.py      # โครง plugin + orchestration
  engine/    detector.py ocr.py provinces.py postprocess.py   # 3 stage
  plugins/   inputs/ filters/ outputs/
  api/       server.py draw.py static/index.html
  cli.py
configs/     default.yaml allowlist.txt
scripts/     download_data.py train.py eval.py synth_province.py
firmware/    esp32cam_streamer/ esp32_gate_mqtt/
```

แผนและรายงาน: [REPORT.md](REPORT.md)
