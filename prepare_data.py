"""เตรียมข้อมูลทั้งหมดที่การทดลองต้องใช้ ด้วยคำสั่งเดียว

    .venv\\Scripts\\python.exe -X utf8 prepare_data.py

ทำ 3 ขั้นตามลำดับ
1. โหลดไฟล์ของ OpenThai2.0-WordSeg-Benchmark ที่เราใช้ ลงโฟลเดอร์ wordseg-bench/
   ได้แก่ ตัวคิดคะแนน (thaiws.py) คะแนนที่ผู้พัฒนาเผยแพร่ (results/expected.json) ไฟล์ NOTICE
   และชุดทดสอบ 4 ชุดที่ licence ให้แจกต่อได้ (TUD, PUD, Wisesight-160, VISTEC) รวมราว 16 MB
2. สร้างข้อความสำหรับจับเวลา data/corpus_1m.txt จากบรรทัดแรก ๆ ของ VISTEC จนครบ 1 ล้านตัวอักษร
3. โหลดส่วนภาษาไทยของ TALPCo แล้วแปลงเป็นชุดทดสอบ data/talpco_tha.jsonl รูปแบบเดียวกับ benchmark

สิ่งที่ตั้งใจไม่ทำ
    ไม่โหลด BEST, ORCHID และชุดทดสอบของ khammoon ซึ่งเป็น licence แบบไม่ให้ใช้เชิงพาณิชย์
    จึงไม่ใช้ benchmark.py ของ benchmark เอง เพราะการเรียก evaluate() จะไปโหลดสามชุดนั้นมาด้วย

ความปลอดภัยและการทำซ้ำ
    ทุกไฟล์โหลดจาก revision ที่ระบุตายตัว และตรวจ SHA-256 ก่อนใช้ ถ้าไฟล์ไม่ตรงจะหยุดทันที
    ไฟล์ที่มีอยู่แล้วและ SHA-256 ตรง จะไม่ถูกโหลดซ้ำ
"""
from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

from common import BENCH, CORPUS, DATA, ROOT, use_utf8_console

# ---------------------------------------------------------------- แหล่งข้อมูล (ระบุ revision ตายตัว)
BENCHMARK_URL = ("https://huggingface.co/datasets/iapp/OpenThai2.0-WordSeg-Benchmark/resolve/"
                 "08fb0d91d870da6ed3acb7bf2a5f676e88697ea0/")
# ไฟล์ของ benchmark ที่เราใช้ -> SHA-256 (ชุดทดสอบทั้ง 4 ตรงกับค่าใน checksums.json ของ benchmark)
BENCHMARK_FILES = {
    "thaiws.py": "1595e07cc1a6cb048a071e5251c1721fcff703d087dbccd7be6d4ee21d665e3a",
    "results/expected.json": "e5038a03a9f1a5b3b75b055aaaaac7c4cdd9dea1c56577927d53b9d72540ccf4",
    "NOTICE": "8a7b3b7f34f096ee3b505878969df0b512a1e066b329fbf70a8110f65e44b24a",
    "data/tud.jsonl": "837ecf03a4ef55cb5c44bfdef794b1ba377656fccaf9446741431ad5fc93d6a3",
    "data/pud.jsonl": "1deee272f6faa1e9428572e4f93000c7e874050f83d8217694f7e15a075284da",
    "data/ws160.jsonl": "f16ea5744619dda6deec62a1976b2253db3a4255108c59d9d5ae98f1b303ed5c",
    "data/vistec.jsonl": "673d4fc87c0491e4f5ec3668c9f2981f249bbab2edcf8345aff34907ad9bae8f",
}

TALPCO_URL = ("https://raw.githubusercontent.com/matbahasa/TALPCo/"
              "eb4746249830e2c0a8b192a464a74616da3e0453/tha/")
# ประโยคดิบ และประโยคที่ตัดคำแล้ว ของ TALPCo ภาษาไทย -> SHA-256
TALPCO_FILES = {
    "data_tha.txt": "54ebac755368ac63a3bd8c150a6185d5fac39cbe186dac1e3398d6bfe69bdf28",
    "data_tha-token.txt": "2aa4b048eed6d49c97854e1bc4ccc102fbb0c3b2149c1d677623cfb8b7cd82d4",
}
TALPCO_RAW = DATA / "talpco"                # ไฟล์ต้นฉบับของ TALPCo
TALPCO_OUT = DATA / "talpco_tha.jsonl"      # ชุดทดสอบที่แปลงแล้ว

CORPUS_CHARS = 1_000_000                    # ขนาดข้อความจับเวลา เท่ากับที่ประกาศของผู้พัฒนาอ้างถึง


# ---------------------------------------------------------------- การโหลดไฟล์
def sha256(path: Path) -> str:
    """คืนค่า SHA-256 ของไฟล์ โดยอ่านทีละ 1 MB เพื่อไม่ต้องโหลดไฟล์ใหญ่เข้าหน่วยความจำทั้งก้อน"""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, dest: Path, expected_sha256: str) -> None:
    """โหลดไฟล์จาก url มาไว้ที่ dest แล้วตรวจ SHA-256

    ถ้า dest มีอยู่แล้วและ SHA-256 ตรง จะข้าม ถ้าโหลดมาแล้วไม่ตรงจะลบทิ้งและหยุดโปรแกรม
    เพราะไฟล์ที่ไม่ตรงอาจเป็นคนละรุ่นกับที่ผลการทดลองนี้ใช้ หรือถูกแก้ระหว่างทาง
    """
    if dest.exists() and sha256(dest) == expected_sha256:
        print(f"  มีแล้ว    {dest.relative_to(ROOT)}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".part")       # โหลดลงไฟล์ชั่วคราวก่อน กันไฟล์ค้างครึ่ง ๆ
    print(f"  กำลังโหลด {dest.relative_to(ROOT)}", flush=True)
    with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as file:
        while chunk := response.read(1 << 20):
            file.write(chunk)
    actual = sha256(partial)
    if actual != expected_sha256:
        partial.unlink()
        raise SystemExit(f"SHA-256 ของ {url} ไม่ตรง (ได้ {actual}) จึงไม่ใช้ไฟล์นี้")
    partial.replace(dest)


def fetch_benchmark() -> None:
    """ขั้นที่ 1: โหลดไฟล์ของ benchmark ที่เราใช้ ลง wordseg-bench/"""
    print("1) OpenThai2.0-WordSeg-Benchmark")
    for name, checksum in BENCHMARK_FILES.items():
        download(BENCHMARK_URL + name, BENCH / name, checksum)


# ---------------------------------------------------------------- ข้อความจับเวลา
def build_corpus() -> None:
    """ขั้นที่ 2: สร้าง data/corpus_1m.txt จากบรรทัดแรก ๆ ของ VISTEC จนครบ CORPUS_CHARS ตัวอักษร

    หนึ่งข้อความของ VISTEC เป็นหนึ่งบรรทัด ขึ้นบรรทัดใหม่ในข้อความถูกแทนด้วยช่องว่าง และข้ามข้อความว่าง
    """
    print("2) ข้อความจับเวลา")
    lines, chars = [], 0
    with (BENCH / "data" / "vistec.jsonl").open(encoding="utf-8") as file:
        for raw in file:
            text = json.loads(raw)["text"].replace("\n", " ").strip()
            if not text:
                continue
            lines.append(text)
            chars += len(text)
            if chars >= CORPUS_CHARS:           # หยุดที่บรรทัดที่ทำให้ครบ ไม่ตัดกลางบรรทัด
                break
    CORPUS.parent.mkdir(parents=True, exist_ok=True)
    CORPUS.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"  {len(lines):,} บรรทัด {chars:,} ตัวอักษร -> {CORPUS.relative_to(ROOT)}")


# ---------------------------------------------------------------- TALPCo
def talpco_sentences() -> dict[str, str]:
    """อ่านประโยคดิบของ TALPCo คืน {รหัสประโยค: ประโยค}

    รูปแบบไฟล์: หนึ่งประโยคต่อบรรทัด เป็น "รหัส<TAB>ประโยค" เข้ารหัส UTF-8 และอาจมี BOM นำหน้า
    """
    sentences = {}
    for line in (TALPCO_RAW / "data_tha.txt").read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            sentence_id, text = line.split("\t", 1)
            sentences[sentence_id.strip()] = text.strip()
    return sentences


def talpco_tokens() -> dict[str, list[str]]:
    """อ่านประโยคที่ตัดคำแล้วของ TALPCo คืน {รหัสประโยค: รายการคำ}

    รูปแบบไฟล์: บรรทัดแรกของแต่ละประโยคคือรหัส ตามด้วยคำบรรทัดละคำ และจบด้วยบรรทัด <EOS>
    """
    tokens, sentence_id, current = {}, None, []
    for line in (TALPCO_RAW / "data_tha-token.txt").read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line:
            continue
        if sentence_id is None:                 # บรรทัดแรกหลังจบประโยคก่อนหน้า คือรหัสของประโยคใหม่
            sentence_id, current = line, []
        elif line == "<EOS>":
            tokens[sentence_id], sentence_id = current, None
        else:
            current.append(line)
    return tokens


def token_spans(text: str, tokens: list[str]) -> list[list[int]] | None:
    """หาตำแหน่ง [เริ่ม, จบ) ของแต่ละคำในประโยคดิบ คืน None ถ้าคำเรียงต่อกันแล้วไม่ตรงกับประโยค

    ช่องว่างในประโยคดิบไม่นับเป็นคำ จึงข้ามก่อนวางคำถัดไป เหมือนที่ benchmark ทำ
    """
    spans, position = [], 0
    for token in tokens:
        while position < len(text) and text[position].isspace():
            position += 1
        if not text.startswith(token, position):
            return None
        spans.append([position, position + len(token)])
        position += len(token)
    return spans if not text[position:].strip() else None       # ต้องไม่เหลือตัวอักษรที่ไม่มีคำครอบ


def build_talpco() -> None:
    """ขั้นที่ 3: โหลด TALPCo ภาษาไทย แล้วแปลงเป็น data/talpco_tha.jsonl

    แต่ละแถวเป็น {"id", "src_id", "text", "gold"} รูปแบบเดียวกับชุดทดสอบของ benchmark
    ประโยคที่คำในไฟล์ตัดคำไม่ตรงกับประโยคดิบ (ต้นฉบับสะกดไม่ตรงกันเอง) จะถูกข้ามและรายงาน
    """
    print("3) TALPCo ภาษาไทย")
    for name, checksum in TALPCO_FILES.items():
        download(TALPCO_URL + name, TALPCO_RAW / name, checksum)
    sentences, tokens = talpco_sentences(), talpco_tokens()
    rows, skipped = [], []
    for sentence_id, text in sentences.items():
        gold = token_spans(text, tokens[sentence_id]) if sentence_id in tokens else None
        if gold is None:
            skipped.append(sentence_id)
            continue
        rows.append({"id": len(rows), "src_id": sentence_id, "text": text, "gold": gold})
    TALPCO_OUT.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                          encoding="utf-8", newline="\n")
    words = sum(len(row["gold"]) for row in rows)
    print(f"  {len(rows):,} ประโยค {words:,} คำ -> {TALPCO_OUT.relative_to(ROOT)}")
    if skipped:
        print(f"  ข้าม {len(skipped)} ประโยคที่คำไม่ตรงกับประโยคดิบ: {', '.join(skipped)}")


def main() -> None:
    """รันทั้ง 3 ขั้นตามลำดับ (ขั้นที่ 2 ใช้ไฟล์ VISTEC จากขั้นที่ 1)"""
    use_utf8_console()
    fetch_benchmark()
    build_corpus()
    build_talpco()


if __name__ == "__main__":
    main()
