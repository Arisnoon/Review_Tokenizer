"""วัดความแม่น (word F1) ของ DeepCut บนชุดทดสอบเดียวกับ run_benchmark.py เพื่อใช้เป็นตัวเทียบ

    .venv-deepcut\\Scripts\\python.exe -X utf8 score_deepcut.py                 # TALPCo, TUD, PUD, Wisesight-160
    .venv-deepcut\\Scripts\\python.exe -X utf8 score_deepcut.py --sets vistec   # ชุดใหญ่ ใช้เวลาราว 20 นาที

ใช้ทำสองอย่าง
1. TALPCo ไม่มีคะแนนของระบบอื่นให้เทียบ จึงวัด DeepCut ไว้เป็นตัวเทียบ
   ควรอ่านคะแนนนี้โดยรู้ว่า DeepCut ได้เปรียบ เพราะเฉลยของ TALPCo เริ่มจากผลของ DeepCut แล้วมีคนตรวจแก้
2. TUD, PUD และ Wisesight-160 มีคะแนนของ DeepCut ที่ผู้พัฒนา benchmark เผยแพร่ไว้
   ถ้าเราวัดได้ตรงกัน แปลว่า DeepCut ที่เราติดตั้งทำงานเหมือนของเขา ซึ่งรองรับการเทียบความเร็วใน bench_deepcut.py

วิธีวัดเหมือน run_benchmark.py คือเรียกทีละบรรทัด แล้วคิดคะแนนด้วยตัวคิดคะแนนของ benchmark

ผลลัพธ์
    results/f1_deepcut.json  ถ้ารันบางชุด จะแก้เฉพาะชุดนั้นในไฟล์ ชุดอื่นที่วัดไว้ก่อนยังอยู่

ต้องรันใน environment ที่มี DeepCut และ TensorFlow (.venv-deepcut) และต้องรัน prepare_data.py ก่อน
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os

from common import (RESULTS, ROOT, TEST_SETS, load_test_set, package_version, published_scores, read_json, score_rows,
                    use_utf8_console, write_json)

OUTPUT = RESULTS / "f1_deepcut.json"
DEFAULT_SETS = ["talpco_tha", "tud", "pud", "ws160"]     # VISTEC ใหญ่เกินกว่าจะรันทุกครั้ง จึงไม่อยู่ในค่าเริ่มต้น
PUBLISHED_NAME = "deepcut-0.7.0.0"                        # ชื่อของ DeepCut ในไฟล์คะแนนที่ผู้พัฒนาเผยแพร่


def quiet_tensorflow() -> None:
    """ลดข้อความของ TensorFlow และให้รันบน CPU ต้องเรียกก่อน import deepcut เพราะ TensorFlow อ่านค่าตอน import"""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")


def main() -> None:
    """อ่านตัวเลือก วัดชุดที่ขอ พิมพ์เทียบกับค่าที่เผยแพร่ แล้วบันทึก results/f1_deepcut.json"""
    use_utf8_console()
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sets", nargs="+", choices=list(TEST_SETS), default=DEFAULT_SETS,
                        help="ชุดทดสอบที่จะวัด")
    args = parser.parse_args()
    quiet_tensorflow()
    import deepcut

    def segment(text: str) -> list[str]:
        """ตัดหนึ่งบรรทัดด้วย DeepCut โดยซ่อนแถบความคืบหน้าที่ Keras พิมพ์ทุกครั้งที่เรียก"""
        with contextlib.redirect_stdout(io.StringIO()):
            return deepcut.tokenize(text)

    published = published_scores()["word_f1"][PUBLISHED_NAME]
    system = f"deepcut {package_version('deepcut')}"
    # เริ่มจากผลที่บันทึกไว้ก่อน เพื่อให้การรันบางชุดไม่ลบผลของชุดอื่น
    scores = read_json(OUTPUT).get("sets", {}) if OUTPUT.exists() else {}
    print(f"{system}: word F1 (เรียกทีละบรรทัด)")
    for name in args.sets:
        rows = load_test_set(name)
        if rows is None:
            raise SystemExit(f"ไม่พบชุดทดสอบ {name}: รัน prepare_data.py ก่อน")
        score = score_rows(rows, segment)
        scores[name] = {"word_f1": score["word_f1"], "boundary_f1": score["boundary_f1"], "lines": score["lines"],
                        "published_word_f1": published.get(name)}
        note = f"(ผู้พัฒนาเผยแพร่ {published[name]:.2f})" if name in published else "(ไม่มีค่าที่เผยแพร่)"
        print(f"  {TEST_SETS[name][0]:7s} word F1 {score['word_f1']:6.2f}  {note}   "
              f"boundary F1 {score['boundary_f1']:6.2f}   {score['lines']:,} บรรทัด {score['seconds']:.0f} วินาที",
              flush=True)
    write_json(OUTPUT, {"system": system, "sets": scores})
    print(f"\nบันทึกผลที่ {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
