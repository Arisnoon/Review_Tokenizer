"""วัดความแม่น (word F1) ของ OpenThai2.0-LATTE-WordSeg-Lite บนเครื่องของเรา

    .venv\\Scripts\\python.exe -X utf8 run_benchmark.py                # ทั้ง 3 ระดับ
    .venv\\Scripts\\python.exe -X utf8 run_benchmark.py --unit short   # ระดับเดียว

ชุดทดสอบ
    4 ชุดของ OpenThai2.0-WordSeg-Benchmark ที่ licence ให้แจกต่อได้ คือ TUD, PUD, Wisesight-160 และ VISTEC
    ค่าเฉลี่ยของ 4 ชุดนี้ benchmark เรียกว่า avg4_commercial และมีค่าที่ผู้พัฒนาเผยแพร่ให้เทียบ
    กับ TALPCo ภาษาไทย ซึ่งเราหามาเพิ่มเอง จึงรายงานแยกและไม่รวมในค่าเฉลี่ย
    BEST, ORCHID และชุดทดสอบของ khammoon ไม่ใช้ เพราะเป็น licence แบบไม่ให้ใช้เชิงพาณิชย์

วิธีวัด
    เรียก tokenizer ทีละบรรทัดบน CPU 1 thread แล้วคิดคะแนนด้วยตัวคิดคะแนนของ benchmark (thaiws.py)
    ผลตัดคำแบบทีละบรรทัดได้เหมือนเดิมทุกครั้ง คะแนนจึงไม่เปลี่ยนเมื่อรันซ้ำ

ผลลัพธ์
    results/f1.json  คะแนนของแต่ละระดับ ทั้งที่เราวัดได้และที่ผู้พัฒนาเผยแพร่
    ถ้ารันระดับเดียว จะแก้เฉพาะระดับนั้นในไฟล์ ระดับอื่นที่วัดไว้ก่อนยังอยู่

ต้องรัน prepare_data.py ก่อน เพื่อให้มีชุดทดสอบและตัวคิดคะแนน
"""
from __future__ import annotations

import argparse
import time

from common import (BENCHMARK_SETS, EXTRA_SETS, RESULTS, ROOT, TEST_SETS, load_test_set, package_version,
                    published_scores, read_json, score_rows, use_utf8_console, write_json)

UNITS = ["long", "short", "minimal"]
OUTPUT = RESULTS / "f1.json"
# ชื่อของ model นี้ในไฟล์คะแนนที่ผู้พัฒนาเผยแพร่ ต่อท้ายด้วยชื่อระดับ
PUBLISHED_NAME = "openthai2.0-latte-wordseg-lite-{unit}"
AVERAGE_KEY = "avg4_commercial"             # ค่าเฉลี่ยของ 4 ชุดที่แจกต่อได้ ในไฟล์ที่ผู้พัฒนาเผยแพร่


def score_unit(unit: str, published: dict) -> dict:
    """วัดคะแนนของหนึ่งระดับบนทุกชุดทดสอบ พิมพ์ผลเทียบกับที่ผู้พัฒนาเผยแพร่ แล้วคืนผลสำหรับบันทึก

    unit:      "long", "short" หรือ "minimal"
    published: คะแนนที่ผู้พัฒนาเผยแพร่ จาก published_scores()
    """
    from iapp_ai.nlp import Tokenizer       # import ที่นี่เพื่อให้ --help ใช้ได้โดยไม่ต้องโหลด model

    tokenizer = Tokenizer(unit=unit)
    name = PUBLISHED_NAME.format(unit=unit)
    published_sets = published["word_f1"][name]
    word, boundary = {}, {}
    print(f"\nระดับ {unit}")

    # 4 ชุดของ benchmark: ทุกชุดมีค่าที่ผู้พัฒนาเผยแพร่ให้เทียบ
    for set_name in BENCHMARK_SETS:
        rows = load_test_set(set_name)
        if rows is None:
            raise SystemExit(f"ไม่พบชุดทดสอบ {set_name}: รัน prepare_data.py ก่อน")
        score = score_rows(rows, tokenizer.word_tokenize)
        word[set_name], boundary[set_name] = score["word_f1"], score["boundary_f1"]
        print(f"  {TEST_SETS[set_name][0]:7s} word F1 {score['word_f1']:6.2f}  "
              f"(ผู้พัฒนาเผยแพร่ {published_sets[set_name]:6.2f})   boundary F1 {score['boundary_f1']:6.2f}   "
              f"{score['lines']:,} บรรทัด {score['seconds']:.0f} วินาที")
    average = round(sum(word.values()) / len(word), 2)
    average_boundary = round(sum(boundary.values()) / len(boundary), 2)
    published_average = published["averages"][name][AVERAGE_KEY]
    print(f"  {'เฉลี่ย':7s} word F1 {average:6.2f}  (ผู้พัฒนาเผยแพร่ {published_average:6.2f})   "
          f"boundary F1 {average_boundary:6.2f}")

    # ชุดที่เราหามาเพิ่ม: ไม่มีค่าที่เผยแพร่ และไม่รวมในค่าเฉลี่ย เพื่อให้ค่าเฉลี่ยยังเทียบกับของผู้พัฒนาได้
    extra = {}
    for set_name in EXTRA_SETS:
        rows = load_test_set(set_name)
        if rows is None:                    # ชุดเสริมไม่บังคับ ถ้าไม่มีไฟล์ก็ข้าม
            continue
        score = score_rows(rows, tokenizer.word_tokenize)
        extra[set_name] = {"word_f1": score["word_f1"], "boundary_f1": score["boundary_f1"], "lines": score["lines"]}
        print(f"  {TEST_SETS[set_name][0]:7s} word F1 {score['word_f1']:6.2f}  (ชุดเพิ่ม ไม่รวมในค่าเฉลี่ย)     "
              f"boundary F1 {score['boundary_f1']:6.2f}   {score['lines']:,} บรรทัด {score['seconds']:.0f} วินาที")

    return {"word_f1": word, "boundary_f1": boundary, "average_word_f1": average,
            "average_boundary_f1": average_boundary,
            "published_word_f1": {set_name: published_sets[set_name] for set_name in BENCHMARK_SETS},
            "published_average_word_f1": published_average, "extra": extra}


def print_leaderboard(units: dict, published: dict, version: str) -> None:
    """พิมพ์ค่าเฉลี่ย 4 ชุดของเรา เรียงรวมกับระบบอื่นที่ผู้พัฒนา benchmark เผยแพร่ไว้

    คะแนนของระบบอื่นไม่ได้วัดซ้ำที่นี่ (ยกเว้น DeepCut ซึ่ง score_deepcut.py วัดซ้ำบางชุด)
    """
    board = [(scores[AVERAGE_KEY], name) for name, scores in published["averages"].items()
             if "(unit rule)" not in name                   # แถวที่เลือกระดับตามชุดทดสอบ ไม่ใช่ระบบจริง
             and not name.startswith("openthai2.0-latte")]  # ของ model นี้ใช้ค่าที่เราวัดเองแทน
    board += [(result["average_word_f1"], f"> iapp-ai {version} {unit} (เราวัดเอง)") for unit, result in units.items()]
    print("\nword F1 เฉลี่ย 4 ชุด เทียบกับระบบอื่น (ค่าของระบบอื่นมาจากผู้พัฒนา benchmark):")
    for value, name in sorted(board, reverse=True):
        print(f"  {value:6.2f}  {name}")


def main() -> None:
    """อ่านตัวเลือก วัดระดับที่ขอ พิมพ์ตารางเทียบ แล้วบันทึก results/f1.json"""
    use_utf8_console()
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--unit", choices=UNITS, help="วัดระดับเดียว (ไม่ระบุ = ทั้ง 3 ระดับ)")
    args = parser.parse_args()

    published = published_scores()
    version = package_version("iapp-ai")
    print(f"iapp-ai {version}: word F1 บนชุดทดสอบที่แจกต่อได้ (CPU 1 thread, เรียกทีละบรรทัด)")
    units = {unit: score_unit(unit, published) for unit in ([args.unit] if args.unit else UNITS)}
    print_leaderboard(units, published, version)

    # รวมกับผลของระดับอื่นที่บันทึกไว้ก่อน เพื่อให้การรันระดับเดียวไม่ลบผลของระดับที่เหลือ
    saved = read_json(OUTPUT).get("units", {}) if OUTPUT.exists() else {}
    saved.update(units)
    write_json(OUTPUT, {"system": f"iapp-ai {version}", "date": time.strftime("%Y-%m-%d"),
                        "sets": BENCHMARK_SETS, "units": {unit: saved[unit] for unit in UNITS if unit in saved}})
    print(f"\nบันทึกผลที่ {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
