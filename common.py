"""ของใช้ร่วมกันของทุกสคริปต์ในโปรเจกต์นี้

รวมสิ่งที่สคริปต์หลายตัวต้องใช้เหมือนกันไว้ที่เดียว ได้แก่
- ตำแหน่งโฟลเดอร์และไฟล์หลัก
- รายชื่อชุดทดสอบ
- การตั้ง console ให้พิมพ์ภาษาไทยได้
- การอ่านข้อความสำหรับจับเวลา และการอ่านเขียนไฟล์ผลแบบ JSON
- การเรียกตัวคิดคะแนนของ benchmark

ไฟล์นี้ถูก import จาก Python หลาย environment (.venv, .venv-dml, .venv-gpu12, .venv-deepcut)
ซึ่งติดตั้ง package ไม่เหมือนกัน ระดับบนสุดของไฟล์จึงใช้เฉพาะ standard library
ส่วน NumPy ที่ตัวคิดคะแนนต้องใช้ จะถูก import เมื่อเรียก load_scorer() เท่านั้น
"""
from __future__ import annotations

import json
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import Callable, Iterable

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"              # ข้อมูลที่ prepare_data.py เตรียมไว้ (ข้อความจับเวลา และ TALPCo)
RESULTS = ROOT / "results"        # ผลการทดลองทั้งหมด
BENCH = ROOT / "wordseg-bench"    # ไฟล์ของ OpenThai2.0-WordSeg-Benchmark ที่ prepare_data.py โหลดมา
CORPUS = DATA / "corpus_1m.txt"   # ข้อความ 1 ล้านตัวอักษรที่ใช้จับเวลา

# ชุดทดสอบ: ชื่อไฟล์ (ไม่มี .jsonl) -> (ชื่อที่แสดง, โฟลเดอร์ที่ไฟล์อยู่)
TEST_SETS = {
    "tud": ("TUD", BENCH / "data"),
    "pud": ("PUD", BENCH / "data"),
    "ws160": ("WS160", BENCH / "data"),
    "vistec": ("VISTEC", BENCH / "data"),
    "talpco_tha": ("TALPCo", DATA),
}
# 4 ชุดของ benchmark ที่ licence ให้แจกต่อได้ ใช้คิดค่าเฉลี่ยเพื่อเทียบกับที่ผู้พัฒนาเผยแพร่
BENCHMARK_SETS = ["tud", "pud", "ws160", "vistec"]
# ชุดที่เราหามาเพิ่มเอง รายงานแยก ไม่รวมในค่าเฉลี่ย
EXTRA_SETS = ["talpco_tha"]

# ชื่อแบบ "เรียกทีละบรรทัด" ที่ bench.py เก็บในไฟล์ผล (แบบ batch ใช้ชื่อเช่น "batch (4,096 chars)")
LINE_MODE = "one line per call"


def mode_label(mode: str) -> str:
    """แปลงชื่อแบบการเรียกที่เก็บในไฟล์ผลของ bench.py (ภาษาอังกฤษ) เป็นข้อความภาษาไทยสำหรับแสดง"""
    if mode == LINE_MODE:
        return "ทีละบรรทัด"
    return mode.replace("(", "").replace(" chars)", " ตัวอักษร")       # "batch (4,096 chars)" -> "batch 4,096 ตัวอักษร"


def use_utf8_console() -> None:
    """ทำให้ print ภาษาไทยได้ทุกเครื่อง

    Windows ที่ภาษาระบบไม่ใช่ภาษาไทย (เช่น code page 1252) จะ error เมื่อ print ภาษาไทยลง pipe หรือไฟล์
    จึงบังคับ stdout และ stderr เป็น UTF-8 ตั้งแต่ต้นโปรแกรม
    """
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):      # stream ที่ถูกแทนด้วยของอื่นอาจไม่มี reconfigure
            stream.reconfigure(encoding="utf-8")


def package_version(name: str) -> str:
    """คืนรุ่นของ package ที่ติดตั้งอยู่ หรือ "-" ถ้าไม่ได้ติดตั้ง"""
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "-"


def read_json(path: Path):
    """อ่านไฟล์ JSON แบบ UTF-8"""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, data) -> None:
    """เขียนไฟล์ JSON แบบ UTF-8 ให้อ่านด้วยตาได้ และสร้างโฟลเดอร์ให้ถ้ายังไม่มี

    ensure_ascii=False เก็บภาษาไทยเป็นตัวอักษรจริง ไม่แปลงเป็น \\uXXXX
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def load_corpus(max_chars: int = 0) -> list[str]:
    """อ่านข้อความจับเวลา (data/corpus_1m.txt) เป็นรายการบรรทัด

    max_chars: ถ้ามากกว่า 0 จะคืนเฉพาะบรรทัดแรก ๆ จนครบจำนวนตัวอักษรนี้ ใช้สำหรับลองรันเร็ว ๆ
    ถ้ายังไม่มีไฟล์ จะหยุดโปรแกรมพร้อมบอกวิธีสร้าง
    """
    if not CORPUS.exists():
        raise SystemExit(f"ไม่พบ {CORPUS}: รัน prepare_data.py ก่อน")
    lines = CORPUS.read_text(encoding="utf-8").splitlines()
    if max_chars <= 0:
        return lines
    kept, total = [], 0
    for line in lines:
        kept.append(line)
        total += len(line)
        if total >= max_chars:              # ครบแล้ว หยุดที่บรรทัดนี้ (ไม่ตัดกลางบรรทัด)
            break
    return kept


def load_test_set(name: str) -> list[dict] | None:
    """อ่านชุดทดสอบตามชื่อใน TEST_SETS คืนรายการแถว {"id", "text", "gold"} หรือ None ถ้ายังไม่มีไฟล์

    gold คือเฉลยที่คนตัดไว้ เก็บเป็นตำแหน่ง [เริ่ม, จบ) ของแต่ละคำในข้อความ
    """
    path = TEST_SETS[name][1] / f"{name}.jsonl"
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def load_scorer():
    """นำตัวคิดคะแนนของ benchmark (wordseg-bench/thaiws.py) เข้ามาใช้ แล้วคืนโมดูลนั้น

    ใช้โค้ดของ benchmark ตรง ๆ โดยไม่แก้ คะแนนที่ได้จึงเทียบกับที่ผู้พัฒนาเผยแพร่ได้
    """
    if not (BENCH / "thaiws.py").exists():
        raise SystemExit(f"ไม่พบ {BENCH / 'thaiws.py'}: รัน prepare_data.py ก่อน")
    if str(BENCH) not in sys.path:
        sys.path.insert(0, str(BENCH))
    import thaiws                           # ต้องมี NumPy ใน environment ที่รัน

    return thaiws


def published_scores() -> dict:
    """คืนคะแนนของทุกระบบที่ผู้พัฒนา benchmark เผยแพร่ไว้ (wordseg-bench/results/expected.json)"""
    path = BENCH / "results" / "expected.json"
    if not path.exists():
        raise SystemExit(f"ไม่พบ {path}: รัน prepare_data.py ก่อน")
    return read_json(path)


def score_rows(rows: list[dict], segment: Callable[[str], Iterable[str]]) -> dict:
    """ให้ segment ตัดข้อความของชุดทดสอบทีละบรรทัด แล้วคิดคะแนนเทียบกับเฉลย

    rows:    แถวของชุดทดสอบจาก load_test_set()
    segment: ฟังก์ชัน ข้อความ -> รายการคำ ของตัวตัดคำที่ต้องการวัด
    คืน {"word_f1", "boundary_f1", "lines", "seconds"}

    word F1 นับคำที่ตำแหน่งเริ่มและจบตรงกับเฉลยพอดี ส่วน boundary F1 นับเฉพาะจุดตัดระหว่างตัวอักษร
    ทั้งสองค่าคิดรวมทั้งชุด (micro) และปัดสองตำแหน่งเหมือนที่ benchmark รายงาน
    """
    thaiws = load_scorer()
    start = time.perf_counter()
    predictions = [{"id": row["id"], "tokens": list(segment(row["text"]))} for row in rows]
    seconds = time.perf_counter() - start
    # word_counts และ boundary_counts คืน (ถูก, ที่ระบบตัด, ในเฉลย) ต่อบรรทัด จึงรวมทุกบรรทัดก่อนคิด F1
    word = thaiws.f1(thaiws.word_counts(rows, predictions).sum(0))
    boundary = thaiws.f1(thaiws.boundary_counts(rows, predictions).sum(0))
    return {"word_f1": round(float(word), 2), "boundary_f1": round(float(boundary), 2),
            "lines": len(rows), "seconds": seconds}
