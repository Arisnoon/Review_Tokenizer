"""วัดหน่วยความจำ (RAM) ที่ process ใช้ เมื่อโหลดและรัน OpenThai2.0-LATTE-WordSeg-Lite

    .venv\\Scripts\\python.exe -X utf8 check_ram.py

ใช้ตรวจคำอ้างว่า "ใช้ RAM 136 MB" โดยวัด 3 แบบ แต่ละแบบรันใน process ใหม่เพื่อไม่ให้ค่าสูงสุดปนกัน
    load      import package โหลด model แล้วตัด 1 ประโยค
    per_line  ตัดข้อความ data/corpus_1m.txt โดยเรียกทีละบรรทัด
    batch     ตัดข้อความเดียวกันโดยส่งทุกบรรทัดในครั้งเดียว

ค่าที่รายงาน (MB) ของแต่ละแบบ
    python_baseline_mb  RAM ของ Python เปล่า ๆ ก่อน import package
    after_load_mb       RAM หลังโหลด model และตัด 1 ประโยค
    end_rss_mb          RAM ตอนจบการทำงาน
    peak_mb             RAM สูงสุดระหว่างการทำงาน (peak working set ของ Windows)

ผู้พัฒนาวัดค่า 136 MB กับข้อความราว 150,000 ตัวอักษรแบบ batch ส่วนแบบ batch ของเราส่ง 1 ล้านตัวอักษร
ในครั้งเดียว ตัวเลขจึงเทียบกันตรง ๆ ไม่ได้

ผลลัพธ์
    results/ram.json

ต้องติดตั้ง psutil ใน environment ที่รัน และต้องรัน prepare_data.py ก่อน
"""
from __future__ import annotations

import json
import subprocess
import sys

from common import RESULTS, ROOT, load_corpus, use_utf8_console, write_json

OUTPUT = RESULTS / "ram.json"
MODES = ["load", "per_line", "batch"]


def megabytes(size: int) -> float:
    """แปลงจำนวน byte เป็น MB ทศนิยมหนึ่งตำแหน่ง"""
    return round(size / 2**20, 1)


def measure(mode: str) -> dict:
    """วัด RAM ของ process นี้ตามแบบที่ขอ ต้องเรียกใน process ใหม่ที่ยังไม่เคย import package

    mode: "load", "per_line" หรือ "batch" (ดูคำอธิบายที่หัวไฟล์)
    """
    import psutil

    process = psutil.Process()
    baseline = megabytes(process.memory_info().rss)         # ก่อน import package และ ONNX Runtime

    from iapp_ai.nlp import Tokenizer

    tokenizer = Tokenizer(unit="short")
    tokenizer.word_tokenize("เจ้าหน้าที่การศึกษาเปิดเผยว่า")
    loaded = megabytes(process.memory_info().rss)
    if mode != "load":
        lines = load_corpus()
        if mode == "per_line":
            for line in lines:
                tokenizer.word_tokenize(line)
        else:
            tokenizer.word_tokenize_batch(lines)
    info = process.memory_info()
    # peak_wset มีเฉพาะบน Windows ระบบอื่นใช้ค่า RSS ตอนจบแทน ซึ่งอาจต่ำกว่าค่าสูงสุดจริง
    return {"mode": mode, "python_baseline_mb": baseline, "after_load_mb": loaded, "end_rss_mb": megabytes(info.rss),
            "peak_mb": megabytes(getattr(info, "peak_wset", info.rss))}


def measure_in_new_process(mode: str) -> dict:
    """เรียกสคริปต์นี้เองใน process ใหม่ให้วัดหนึ่งแบบ แล้วอ่านผลจากบรรทัดสุดท้ายที่มันพิมพ์เป็น JSON

    ต้องใช้ process ใหม่ เพราะค่าสูงสุดของ RAM เป็นค่าสะสมตลอดอายุของ process
    ถ้าวัดหลายแบบใน process เดียว แบบหลังจะได้ค่าสูงสุดของแบบก่อนติดมาด้วย
    """
    done = subprocess.run([sys.executable, "-X", "utf8", __file__, mode], capture_output=True, text=True,
                          encoding="utf-8", check=True)
    return json.loads(done.stdout.strip().splitlines()[-1])


def main() -> None:
    """ถ้าถูกเรียกพร้อมชื่อแบบ จะวัดแบบนั้นแล้วพิมพ์ JSON (ใช้โดย process ลูก) ไม่อย่างนั้นวัดทุกแบบและบันทึกผล"""
    use_utf8_console()
    if len(sys.argv) == 2 and sys.argv[1] in MODES:
        print(json.dumps(measure(sys.argv[1])))
        return

    rows = []
    print(f"{'แบบ':10s} {'Python เปล่า':>12s} {'หลังโหลด model':>15s} {'ตอนจบ':>8s} {'สูงสุด':>8s}   (MB)")
    for mode in MODES:
        row = measure_in_new_process(mode)
        rows.append(row)
        print(f"{mode:10s} {row['python_baseline_mb']:12.1f} {row['after_load_mb']:15.1f} "
              f"{row['end_rss_mb']:8.1f} {row['peak_mb']:8.1f}")
    write_json(OUTPUT, rows)
    print(f"\nบันทึกผลที่ {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
