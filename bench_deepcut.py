"""จับเวลา DeepCut บนเครื่องเดียวกัน เพื่อตรวจคำอ้างว่า LATTE "เร็วกว่า DeepCut 23 เท่า"

    .venv-deepcut\\Scripts\\python.exe -X utf8 bench_deepcut.py                  # ข้อความเต็ม 1 ล้านตัวอักษร (ราว 6 นาที)
    .venv-deepcut\\Scripts\\python.exe -X utf8 bench_deepcut.py --chars 20000    # ลองเร็ว ๆ ไม่บันทึกผล

เงื่อนไขเดียวกับที่ผู้พัฒนาระบุ คือ CPU 1 thread และเรียกทีละบรรทัด (deepcut.tokenize หนึ่งครั้งต่อหนึ่งบรรทัด)
ใช้ข้อความเดียวกับ bench.py (data/corpus_1m.txt) เวลาโหลด model ไม่นับรวม

DeepCut รันช้า จึงวัดรอบเดียวโดยไม่รันซ้ำ ระหว่างรันจะพิมพ์ความเร็วสะสมทุก 500 บรรทัด
ซึ่งใช้ดูได้ว่าความเร็วนิ่งตลอดการรันหรือไม่

ผลลัพธ์
    results/deepcut.json  เวลาและความเร็วของ DeepCut
    ถ้ามี results/cpu.json ของ bench.py อยู่ จะพิมพ์ด้วยว่า LATTE เร็วกว่ากี่เท่า (ไม่ได้บันทึกลงไฟล์)

ต้องรันใน environment ที่มี DeepCut และ TensorFlow (.venv-deepcut) และต้องรัน prepare_data.py ก่อน
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import time

from common import (LINE_MODE, RESULTS, ROOT, load_corpus, mode_label, package_version, read_json, use_utf8_console,
                    write_json)

OUTPUT = RESULTS / "deepcut.json"
WARMUP_LINES = 20           # จำนวนบรรทัดที่รันทิ้งก่อนจับเวลา
PROGRESS_EVERY = 500        # พิมพ์ความเร็วสะสมทุกกี่บรรทัด


def limit_to_one_thread() -> None:
    """บังคับให้ TensorFlow ใช้ CPU 1 thread ต้องเรียกก่อน import tensorflow เพราะค่าพวกนี้ถูกอ่านตอน import

    ใช้ 1 thread เพื่อให้เทียบกับ LATTE ซึ่ง package ตั้งให้ใช้ 1 thread อยู่แล้ว ได้อย่างเป็นธรรม
    """
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")      # ซ่อนข้อความแจ้งทั่วไปของ TensorFlow
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")     # ไม่ใช้การ์ดจอ
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
    os.environ["TF_NUM_INTEROP_THREADS"] = "1"


def time_deepcut(tokenize, lines: list[str]) -> float:
    """เรียก tokenize ทีละบรรทัดกับทุกบรรทัด แล้วคืนเวลารวม (วินาที)

    แถบความคืบหน้าที่ Keras พิมพ์ทุกครั้งที่เรียกถูกซ่อนไว้ เพราะการพิมพ์ลง console นับหมื่นบรรทัดกินเวลาเอง
    """
    done = 0
    start = time.perf_counter()
    for index, line in enumerate(lines, 1):
        with contextlib.redirect_stdout(io.StringIO()):
            tokenize(line)
        done += len(line)
        if index % PROGRESS_EVERY == 0:
            elapsed = time.perf_counter() - start
            print(f"  {index:,} บรรทัด {done:,} ตัวอักษร {elapsed:6.0f} วินาที  "
                  f"({done / elapsed:,.0f} ตัวอักษร/วินาที)", flush=True)
    return time.perf_counter() - start


def print_latte_comparison(deepcut_speed: float) -> None:
    """พิมพ์ว่า LATTE (จาก results/cpu.json ของ bench.py) เร็วกว่า DeepCut กี่เท่า ถ้ายังไม่มีไฟล์นั้นก็ข้าม"""
    latte_path = RESULTS / "cpu.json"
    if not latte_path.exists():
        return
    for run in read_json(latte_path)["runs"]:
        ratio = run["chars_per_second"] / deepcut_speed
        print(f"LATTE {mode_label(run['mode']):<24} {run['chars_per_second']:>8,.0f} ตัวอักษร/วินาที   "
              f"เร็วกว่า DeepCut {ratio:.1f} เท่า")


def main() -> None:
    """อ่านตัวเลือก จับเวลา DeepCut พิมพ์ผลเทียบกับ LATTE แล้วบันทึก results/deepcut.json"""
    use_utf8_console()
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--chars", type=int, default=0,
                        help="ลองกับข้อความแค่จำนวนตัวอักษรนี้ และไม่บันทึกผล (0 = ทั้งไฟล์)")
    args = parser.parse_args()
    lines = load_corpus(args.chars)
    chars = sum(map(len, lines))

    limit_to_one_thread()
    import tensorflow as tf

    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    import deepcut

    # การเรียกครั้งแรกโหลด model จึงจับเวลาแยกไว้ และรันทิ้งอีกเล็กน้อยก่อนเริ่มจับเวลาจริง
    start = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        deepcut.tokenize("เจ้าหน้าที่การศึกษาเปิดเผยว่า")
        load_seconds = time.perf_counter() - start
        for line in lines[:WARMUP_LINES]:
            deepcut.tokenize(line)

    system = f"deepcut {package_version('deepcut')}"
    print(f"{system}, TensorFlow {tf.__version__}, 1 thread, เรียกทีละบรรทัด")
    print(f"ข้อความ   {len(lines):,} บรรทัด {chars:,} ตัวอักษร   (โหลด model {load_seconds:.1f} วินาที)", flush=True)
    seconds = time_deepcut(deepcut.tokenize, lines)
    speed = chars / seconds
    print(f"\nDeepCut   {seconds:7.1f} วินาที   {speed:>8,.0f} ตัวอักษร/วินาที   "
          f"= {seconds * 1e6 / chars:.0f} วินาทีต่อ 1 ล้านตัวอักษร")
    print_latte_comparison(speed)

    if args.chars:
        print("\nลองกับข้อความบางส่วน จึงไม่บันทึกผล")
        return
    write_json(OUTPUT, {"system": system, "tensorflow": tf.__version__, "threads": 1, "mode": LINE_MODE,
                        "lines": len(lines), "chars": chars, "seconds": seconds, "chars_per_second": speed,
                        "seconds_per_million_chars": seconds * 1e6 / chars, "load_seconds": load_seconds})
    print(f"\nบันทึกผลที่ {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
