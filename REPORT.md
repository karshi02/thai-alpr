# รายงานวางแผนระบบตรวจจับป้ายทะเบียนรถไทย (Thai ALPR)

วันที่: 2026-09-22
สถานะ: วางแผนเสร็จ ยังไม่เริ่มโค้ด

## สรุป

ทำได้บนเครื่องปัจจุบัน (GTX 1650 4GB) ภายใน 4 สัปดาห์ ใช้ pipeline 3 stage: YOLOv8n ตรวจกรอบป้าย -> YOLO ตรวจตัวอักษรบรรทัดบน -> classifier จังหวัด 77 class
ไม่มีโปรเจก open source ไทยที่ใช้ได้ทันที ต้องประกอบจาก dataset Roboflow + โครง pipeline จาก ANPR ทั่วไป

## สิ่งที่ค้นพบ

| หัวข้อ | ผล |
|---|---|
| โปรเจกไทยที่มีอยู่ | 2 repo (OpenCV เก่า, tiny-YOLO+KNN) + 1 paper (YOLOv10+Tesseract 50k รูป) ใช้เป็นแนวทาง ไม่ใช่ตัวจบ |
| Dataset | Roboflow มี 5 ชุด: กรอบป้าย ~400 รูป, ตัวอักษร 2.5k รูป, จังหวัด 211 รูป (น้อยที่สุด ต้อง synthetic เพิ่ม) |
| เครื่อง | Python 3.14 บนเครื่อง torch ไม่รองรับ ต้องใช้ 3.11 ใน venv แยก |
| จุดยาก | ตัวอักษรไทย, จังหวัด 77 ชื่อฟอนต์เล็ก, ป้ายหลายสี/กราฟิก, กลางคืน/ฝน |

## แผน 4 สัปดาห์

| สัปดาห์ | ส่งมอบ |
|---|---|
| 1 (22-28 ก.ย.) | env + Stage 1 ตรวจกรอบป้าย mAP50 > 0.9 |
| 2 (29 ก.ย.-5 ต.ค.) | Stage 2 อ่านบรรทัดบน end-to-end |
| 3 (6-12 ต.ค.) | Stage 3 จังหวัด + ประเภทป้ายจากสี |
| 4 (13-20 ต.ค.) | FastAPI `/detect` + วิดีโอ tracking + วัดผล + ONNX |

## เป้าหมาย MVP

- ทะเบียนถูกทั้ง string > 80%, จังหวัดถูก > 85%
- Latency < 100 ms/เฟรม บน GTX 1650
- วัดบน test set 200 รูปจริง (กลางวัน 60% / คืน 40%) แยกจาก train

## ความเสี่ยงหลัก

1. Dataset จังหวัดน้อย -> render synthetic จาก font + เก็บรูปจริงเพิ่ม
2. VRAM 4GB -> yolov8n, batch 8, amp
3. License dataset Roboflow ไม่ชัด -> เช็คก่อนใช้เชิงพาณิชย์

## เอกสารที่สร้าง

- Claude Docs (แผนฉบับเต็ม แก้/comment ได้): https://claude.ai/code/artifact/3d21fd26-12b2-40a3-8e38-fadb10a2aeb8
- Obsidian vault: `Obsidian Vault/ThaiALPR/` 7 โน้ต เริ่มที่ `ThaiALPR — ภาพรวมโปรเจค`

## ขั้นถัดไป

- [ ] สมัคร Roboflow เอา API key
- [ ] ลง Python 3.11 + CUDA
- [ ] สร้างโครง repo `data/ models/ src/ api/ scripts/`
- [ ] เริ่มสัปดาห์ 1: train Stage 1
