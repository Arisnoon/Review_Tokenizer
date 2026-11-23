"""ดูผลตัดคำของ OpenThai2.0-LATTE-WordSeg-Lite ด้วยตา: ตัดได้ 3 ระดับ และขอดูได้หลายคำตอบ

    .venv\\Scripts\\python.exe -X utf8 review.py

ใช้ package ตามที่ติดตั้งด้วย `pip install "iapp-ai[nlp]"` โดยไม่ตั้งค่าอะไรเพิ่ม (รันบน CPU)
สคริปต์นี้พิมพ์ผลอย่างเดียว ไม่บันทึกไฟล์ ตัวอย่างในตารางของ blog/results.html มาจากผลที่พิมพ์ที่นี่

ส่วนที่ 1  ตัดประโยคเดียวกันด้วย unit ทั้ง 3 ระดับ เพื่อให้เห็นว่าต่างกันตรงไหน
ส่วนที่ 2  ขอ 3 คำตอบแรก (n-best) ของข้อความกำกวม พร้อมความน่าจะเป็นที่ model ให้
"""
from __future__ import annotations

from iapp_ai.nlp import word_tokenize, word_tokenize_nbest

from common import use_utf8_console

# ระดับการตัด เรียงจากหยาบไปละเอียด: (ชื่อที่ส่งให้ package, คำอธิบาย)
UNITS = [
    ("long", "คำประสม: keyword, ชื่อเฉพาะ, ศัพท์เทคนิค"),
    ("short", "คำ (ค่าเริ่มต้น): งาน NLP ทั่วไป"),
    ("minimal", "หน่วยคำ: search ที่อยากได้ผลครอบคลุม"),
]

# ประโยคแรกมาจากประกาศของผู้พัฒนา ที่เหลือเราคิดเอง
# สองประโยคสุดท้ายแสดงว่าระดับ long ไม่ได้รวมคำประสมให้ทุกคำ
SENTENCES = [
    "เจ้าหน้าที่การศึกษาเปิดเผยว่า",
    "ร้านอาหารตามสั่งหน้าปากซอยอร่อยมาก",
    "แม่น้ำเจ้าพระยาไหลผ่านกรุงเทพมหานคร",
    "ฉันซื้อโทรศัพท์มือถือรุ่นใหม่ที่ห้างสรรพสินค้า",
    "โรงพยาบาลจุฬาลงกรณ์เปิดรับสมัครพยาบาลวิชาชีพ",
    "ผู้ว่าราชการจังหวัดเชียงใหม่ตรวจโรงพยาบาลสนาม",
]

# สองข้อความแรกมาจากประกาศของผู้พัฒนา ที่เหลือเราคิดเอง
AMBIGUOUS = ["ตากลม", "ไปหามเหสี", "แปลกว่า", "รอบอก", "ยานี้กินแล้วแพ้"]
NBEST = 3                                   # จำนวนคำตอบที่ขอต่อข้อความ


def heading(title: str) -> None:
    """พิมพ์หัวข้อของแต่ละส่วน คั่นด้วยเส้นให้อ่านง่ายใน terminal"""
    print("=" * 72)
    print(title)
    print("=" * 72)


def joined(words: list[str]) -> str:
    """แสดงรายการคำเป็นบรรทัดเดียว คั่นด้วยขีดตั้ง เพื่อให้เห็นรอยตัดชัด"""
    return " | ".join(words)


def show_units() -> None:
    """ส่วนที่ 1: ตัดแต่ละประโยคด้วยทั้ง 3 ระดับ แล้วพิมพ์ผลกับจำนวนชิ้นเทียบกัน"""
    heading("1. ตัดคำ 3 ระดับ (unit) จาก model เดียว")
    for unit, meaning in UNITS:
        print(f"  {unit:<8} {meaning}")
    for sentence in SENTENCES:
        print(f"\n  {sentence}")
        for unit, _ in UNITS:
            words = word_tokenize(sentence, unit=unit)      # model โหลดครั้งแรกที่เรียก แล้วใช้ซ้ำ
            print(f"    {unit:<8} ({len(words):>2} ชิ้น)  {joined(words)}")


def show_nbest() -> None:
    """ส่วนที่ 2: ขอหลายคำตอบของข้อความกำกวม เรียงจากความน่าจะเป็นมากไปน้อย

    ความน่าจะเป็นใช้จัดอันดับคำตอบเท่านั้น package ระบุว่าไม่ได้ปรับเทียบให้เป็นค่าจริง
    """
    heading("2. n-best: หลายคำตอบของข้อความกำกวม เรียงตามความน่าจะเป็น")
    for phrase in AMBIGUOUS:
        print(f"\n  {phrase}")
        for rank, (words, probability) in enumerate(word_tokenize_nbest(phrase, n=NBEST), 1):
            print(f"    {rank}. {probability:6.1%}  {joined(words)}")


def main() -> None:
    """พิมพ์ทั้งสองส่วนต่อกัน"""
    use_utf8_console()
    show_units()
    print()
    show_nbest()


if __name__ == "__main__":
    main()
