"""ตรวจคำอ้างข้อย่อยของ OpenThai2.0-LATTE-WordSeg-Lite ที่ไม่ใช่ความแม่นหรือความเร็ว

    .venv\\Scripts\\python.exe -X utf8 check_claims.py

ตรวจ 3 ข้อ
1. ขนาดไฟล์    ประกาศบอกว่า model กับ dictionary รวม 3.8 MB จึงดูขนาดไฟล์จริงที่ติดตั้งมากับ package
2. รัน offline  ประกาศบอกว่ารันได้โดยไม่ต้องต่ออินเทอร์เน็ต จึงปิดการเชื่อมต่อเครือข่ายทั้งหมดของ process
               แล้วลองตัดคำและขอหลายคำตอบ
3. ผล batch    model card บอกว่าการส่งเป็น batch อาจตัดต่างจากการเรียกทีละบรรทัดได้ถึง 2.9% ของบรรทัด
               (เพราะ scale ของ int8 คำนวณต่อ batch) จึงนับจำนวนบรรทัดที่ต่างกันบนข้อความ data/corpus_1m.txt
               และตรวจด้วยว่าการเรียกทีละบรรทัดซ้ำได้ผลเหมือนเดิม

ผลลัพธ์
    results/claims.json

ต้องรัน prepare_data.py ก่อน เพื่อให้มีข้อความสำหรับข้อ 3
"""
from __future__ import annotations

import socket
from pathlib import Path

from common import RESULTS, ROOT, load_corpus, use_utf8_console, write_json

OUTPUT = RESULTS / "claims.json"
REPEAT_LINES = 500          # จำนวนบรรทัดที่ใช้ตรวจว่าการเรียกทีละบรรทัดซ้ำได้ผลเหมือนเดิม


def model_size() -> dict:
    """คืนขนาดไฟล์ของ model ที่ติดตั้งมากับ package และผลรวมของ model กับ dictionary (MB, 1 MB = 1,000,000 byte)"""
    import iapp_ai.nlp as nlp

    assets = Path(nlp.__file__).parent / "data" / "latte"
    files = {path.name: path.stat().st_size for path in sorted(assets.iterdir()) if path.is_file()}
    # ประกาศนับเฉพาะ model กับ dictionary ไม่รวมตารางชนิดตัวอักษรและ manifest ซึ่งเล็กมาก
    core = files.get("model.int8.onnx", 0) + files.get("lexicon.txt.gz", 0)
    return {"files_bytes": files, "model_and_dictionary_mb": round(core / 1e6, 2)}


def runs_offline() -> bool:
    """ปิดการเชื่อมต่อเครือข่ายของ process นี้ แล้วลองตัดคำและขอหลายคำตอบ คืน True ถ้าทำได้ครบ

    แทนที่ฟังก์ชันเชื่อมต่อของ socket ด้วยฟังก์ชันที่ error ทันที ถ้า package พยายามต่อเน็ตตรงไหน จะล้มให้เห็น
    การปิดนี้มีผลจนจบ process จึงเรียกเป็นขั้นแรกก่อนที่ package จะโหลด model
    """
    def blocked(*args, **kwargs):
        """ใช้แทนการเชื่อมต่อทุกแบบ เพื่อจำลองเครื่องที่ไม่มีอินเทอร์เน็ต"""
        raise OSError("ปิดเครือข่ายไว้สำหรับการทดสอบนี้")

    socket.socket.connect = blocked
    socket.create_connection = blocked
    from iapp_ai.nlp import word_tokenize, word_tokenize_nbest

    words = word_tokenize("เจ้าหน้าที่การศึกษาเปิดเผยว่า", unit="long")
    readings = word_tokenize_nbest("ตากลม", 3)
    return bool(words) and len(readings) == 3


def batch_versus_line() -> dict:
    """นับบรรทัดที่การส่งเป็น batch ตัดต่างจากการเรียกทีละบรรทัด และตรวจว่าการเรียกทีละบรรทัดซ้ำได้ผลเดิม

    ใช้ระดับ short และขนาด batch เริ่มต้นของ package
    """
    from iapp_ai.nlp import Tokenizer

    lines = load_corpus()
    tokenizer = Tokenizer(unit="short")
    per_line = [tokenizer.word_tokenize(line) for line in lines]
    batch = tokenizer.word_tokenize_batch(lines)
    differ = sum(a != b for a, b in zip(per_line, batch))
    again = [tokenizer.word_tokenize(line) for line in lines[:REPEAT_LINES]]
    return {"lines": len(lines), "differ": differ, "per_line_repeatable": again == per_line[:REPEAT_LINES]}


def main() -> None:
    """รันการตรวจทั้ง 3 ข้อ พิมพ์ผล แล้วบันทึก results/claims.json"""
    use_utf8_console()
    offline = runs_offline()                # ต้องมาก่อน เพื่อให้การโหลด model ครั้งแรกเกิดตอนที่ปิดเครือข่ายแล้ว
    size = model_size()
    comparison = batch_versus_line()

    print("1) ขนาดไฟล์")
    for name, size_bytes in size["files_bytes"].items():
        print(f"   {name:20s} {size_bytes / 1e6:6.2f} MB")
    print(f"   model กับ dictionary  {size['model_and_dictionary_mb']:.2f} MB")
    print(f"2) รันได้เมื่อปิดเครือข่าย: {'ได้' if offline else 'ไม่ได้'}")
    share = comparison["differ"] / comparison["lines"]
    print(f"3) batch ตัดต่างจากทีละบรรทัด {comparison['differ']:,} จาก {comparison['lines']:,} บรรทัด ({share:.2%})")
    print(f"   เรียกทีละบรรทัดซ้ำได้ผลเหมือนเดิม: {'ใช่' if comparison['per_line_repeatable'] else 'ไม่ใช่'}")

    write_json(OUTPUT, {"model_and_dictionary_mb": size["model_and_dictionary_mb"],
                        "files_bytes": size["files_bytes"], "runs_offline": offline,
                        "batch_versus_line": comparison})
    print(f"\nบันทึกผลที่ {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
