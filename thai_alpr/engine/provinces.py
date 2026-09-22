"""77 Thai provinces as printed on plates + fuzzy matcher (Stage 3 fallback)."""
from __future__ import annotations

from rapidfuzz import fuzz, process

PROVINCES: list[str] = [
    "กรุงเทพมหานคร", "กระบี่", "กาญจนบุรี", "กาฬสินธุ์", "กำแพงเพชร", "ขอนแก่น", "จันทบุรี",
    "ฉะเชิงเทรา", "ชลบุรี", "ชัยนาท", "ชัยภูมิ", "ชุมพร", "เชียงราย", "เชียงใหม่", "ตรัง",
    "ตราด", "ตาก", "นครนายก", "นครปฐม", "นครพนม", "นครราชสีมา", "นครศรีธรรมราช", "นครสวรรค์",
    "นนทบุรี", "นราธิวาส", "น่าน", "บึงกาฬ", "บุรีรัมย์", "ปทุมธานี", "ประจวบคีรีขันธ์",
    "ปราจีนบุรี", "ปัตตานี", "พระนครศรีอยุธยา", "พะเยา", "พังงา", "พัทลุง", "พิจิตร",
    "พิษณุโลก", "เพชรบุรี", "เพชรบูรณ์", "แพร่", "ภูเก็ต", "มหาสารคาม", "มุกดาหาร",
    "แม่ฮ่องสอน", "ยโสธร", "ยะลา", "ร้อยเอ็ด", "ระนอง", "ระยอง", "ราชบุรี", "ลพบุรี",
    "ลำปาง", "ลำพูน", "เลย", "ศรีสะเกษ", "สกลนคร", "สงขลา", "สตูล", "สมุทรปราการ",
    "สมุทรสงคราม", "สมุทรสาคร", "สระแก้ว", "สระบุรี", "สิงห์บุรี", "สุโขทัย", "สุพรรณบุรี",
    "สุราษฎร์ธานี", "สุรินทร์", "หนองคาย", "หนองบัวลำภู", "อ่างทอง", "อำนาจเจริญ",
    "อุดรธานี", "อุตรดิตถ์", "อุทัยธานี", "อุบลราชธานี",
]
assert len(PROVINCES) == 77


def match_province(text: str, min_score: int = 60) -> tuple[str, float]:
    """Fuzzy-match noisy OCR text to a province. Returns (province, score 0-1) or ("", 0)."""
    text = (text or "").strip().replace(" ", "")
    if not text:
        return "", 0.0
    hit = process.extractOne(text, PROVINCES, scorer=fuzz.WRatio)
    if hit and hit[1] >= min_score:
        return hit[0], hit[1] / 100.0
    return "", 0.0


# English labels used by the Roboflow province dataset -> Thai name on the plate
EN_TO_TH: dict[str, str] = {
    "bangkok": "กรุงเทพมหานคร", "krabi": "กระบี่", "kanchanaburi": "กาญจนบุรี", "kalasin": "กาฬสินธุ์",
    "kamphaeng phet": "กำแพงเพชร", "khon kaen": "ขอนแก่น", "chanthaburi": "จันทบุรี", "chachoengsao": "ฉะเชิงเทรา",
    "chonburi": "ชลบุรี", "chai nat": "ชัยนาท", "chaiyaphum": "ชัยภูมิ", "chumphon": "ชุมพร", "chiang rai": "เชียงราย",
    "chiang mai": "เชียงใหม่", "trang": "ตรัง", "trat": "ตราด", "tak": "ตาก", "nakhon nayok": "นครนายก",
    "nakhon pathom": "นครปฐม", "nakhon phanom": "นครพนม", "nakhon ratchasima": "นครราชสีมา",
    "nakhon si thammarat": "นครศรีธรรมราช", "nakhon sawan": "นครสวรรค์", "nonthaburi": "นนทบุรี",
    "narathiwat": "นราธิวาส", "nan": "น่าน", "bueng kan": "บึงกาฬ", "buriram": "บุรีรัมย์", "pathum thani": "ปทุมธานี",
    "prachuap khiri khan": "ประจวบคีรีขันธ์", "prachinburi": "ปราจีนบุรี", "pattani": "ปัตตานี",
    "phra nakhon si ayutthaya": "พระนครศรีอยุธยา", "phayao": "พะเยา", "phang nga": "พังงา", "phatthalung": "พัทลุง",
    "phichit": "พิจิตร", "phitsanulok": "พิษณุโลก", "phetchaburi": "เพชรบุรี", "phetchabun": "เพชรบูรณ์", "phrae": "แพร่",
    "phuket": "ภูเก็ต", "maha sarakham": "มหาสารคาม", "mukdahan": "มุกดาหาร", "mae hong son": "แม่ฮ่องสอน",
    "yasothon": "ยโสธร", "yala": "ยะลา", "roi et": "ร้อยเอ็ด", "ranong": "ระนอง", "rayong": "ระยอง", "ratchaburi": "ราชบุรี",
    "lopburi": "ลพบุรี", "lampang": "ลำปาง", "lamphun": "ลำพูน", "loei": "เลย", "sisaket": "ศรีสะเกษ", "sakon nakhon": "สกลนคร",
    "songkhla": "สงขลา", "satun": "สตูล", "samut prakan": "สมุทรปราการ", "samut songkhram": "สมุทรสงคราม",
    "samut sakhon": "สมุทรสาคร", "sa kaeo": "สระแก้ว", "saraburi": "สระบุรี", "sing buri": "สิงห์บุรี", "sukhothai": "สุโขทัย",
    "suphan buri": "สุพรรณบุรี", "surat thani": "สุราษฎร์ธานี", "surin": "สุรินทร์", "nong khai": "หนองคาย",
    "nong bua lamphu": "หนองบัวลำภู", "ang thong": "อ่างทอง", "amnat charoen": "อำนาจเจริญ", "udon thani": "อุดรธานี",
    "uttaradit": "อุตรดิตถ์", "uthai thani": "อุทัยธานี", "ubon ratchathani": "อุบลราชธานี",
}


def en_to_th(label: str) -> str:
    return EN_TO_TH.get(label.strip().lower(), label)
