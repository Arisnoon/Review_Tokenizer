"""สร้างหน้า blog/results.html ซึ่งรวมผลการทดลองทั้งหมดเป็นตารางที่คัดลอกไปวางที่อื่นได้

    .venv\\Scripts\\python.exe -X utf8 blog\\make_tables.py

ทำไมต้องมีสคริปต์นี้
    ตารางเดียวกันต้องใช้สองรูปแบบ คือ ตาราง HTML สำหรับวางในเว็บหรือเอกสารที่รองรับตาราง
    และ Markdown สำหรับเว็บที่เขียนด้วย Markdown การเก็บเนื้อหาของตารางไว้ที่เดียวในไฟล์นี้
    แล้วสร้างทั้งสองรูปแบบจากที่เดียว ทำให้ตัวเลขตรงกันเสมอ

หน้าที่ได้
    แต่ละตารางมีปุ่ม "คัดลอกตาราง" (คัดลอกเป็นตารางจริง) และ "คัดลอกเป็น Markdown"
    หน้าใช้สีดำบนพื้นขาวล้วน เพราะ browser คัดลอกสีและเส้นติดไปกับตารางด้วย

ที่มาของตัวเลข
    ตัวเลขในไฟล์นี้คัดมาจากไฟล์ผลใน results/ ของการทดลองวันที่ 23 พฤศจิกายน 2026
    แต่ละตารางมี comment บอกว่ามาจากไฟล์ไหน ถ้ารันการทดลองใหม่แล้วตัวเลขเปลี่ยน ต้องแก้ตารางที่นี่ด้วย
    ตารางที่เป็นคำอธิบายหรือข้อสรุป (ความหมายของ 3 ระดับ, เทียบกับประกาศ) เขียนขึ้นเอง ไม่ได้มาจากไฟล์ผล

สคริปต์นี้ใช้เฉพาะ standard library จึงรันด้วย Python ตัวไหนก็ได้
"""
from __future__ import annotations

import html
import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAGE = HERE / "results.html"

PAGE_CSS = """
body { max-width: 900px; margin: 32px auto; padding: 0 16px; font-family: "Leelawadee UI", "Segoe UI", Tahoma, sans-serif;
       font-size: 16px; line-height: 1.6; color: #000000; background: #ffffff; }
h1 { font-size: 24px; margin-bottom: 4px; }
h2 { font-size: 18px; margin: 34px 0 6px; }
.sub { margin-top: 0; }
.howto { border: 1px dashed #999999; padding: 8px 14px; font-size: 14px; }
table { border-collapse: collapse; }
th, td { border: 1px solid #999999; padding: 6px 10px; vertical-align: top; text-align: left; }
.note { font-size: 14px; margin: 6px 0 0; }
.bar { margin: 8px 0 0; }
button { font: inherit; font-size: 14px; padding: 4px 12px; margin-right: 6px; cursor: pointer; }
.done { font-size: 14px; margin-left: 4px; }
"""

# ปุ่มคัดลอก: เลือกตารางแล้วสั่ง copy เหมือนผู้ใช้ลากคลุมเอง จึงได้ตารางจริงติด clipboard
# ส่วน Markdown เก็บไว้ใน <textarea hidden> ของแต่ละตาราง แล้วคัดลอกเป็นข้อความ
PAGE_JS = """
function copyNode(node) {
  var selection = window.getSelection(), range = document.createRange(), ok = false;
  range.selectNode(node);
  selection.removeAllRanges();
  selection.addRange(range);
  try { ok = document.execCommand('copy'); } catch (error) { ok = false; }
  selection.removeAllRanges();
  return ok;
}
function copyText(text) {
  var area = document.createElement('textarea'), ok = false;
  area.value = text;
  area.style.position = 'fixed';
  area.style.opacity = '0';
  document.body.appendChild(area);
  area.select();
  try { ok = document.execCommand('copy'); } catch (error) { ok = false; }
  document.body.removeChild(area);
  return ok;
}
document.querySelectorAll('button[data-copy]').forEach(function (button) {
  button.addEventListener('click', function () {
    var section = button.closest('section'), done = section.querySelector('.done');
    var ok = button.dataset.copy === 'table'
      ? copyNode(section.querySelector('table'))
      : copyText(section.querySelector('textarea').value);
    done.textContent = ok ? 'คัดลอกแล้ว' : 'คัดลอกไม่สำเร็จ ให้ลากคลุมตารางแล้วกด Ctrl+C';
    setTimeout(function () { done.textContent = ''; }, 3000);
  });
});
document.documentElement.setAttribute('data-ready', document.querySelectorAll('button[data-copy]').length);
"""


@dataclass(frozen=True)
class Table:
    """เนื้อหาของหนึ่งตาราง ทุกช่องเป็นข้อความ

    title:     หัวข้อที่แสดงเหนือตาราง
    head:      ข้อความของหัวคอลัมน์
    rows:      แถวข้อมูล แต่ละแถวมีจำนวนช่องเท่ากับ head
    numeric:   ลำดับคอลัมน์ (เริ่มจาก 0) ที่เป็นตัวเลข จะจัดชิดขวา
    bold_rows: ลำดับแถว (เริ่มจาก 0) ที่ต้องการเน้นด้วยตัวหนา
    note:      หมายเหตุใต้ตาราง
    """
    title: str
    head: list[str]
    rows: list[list[str]]
    numeric: tuple[int, ...] = ()
    bold_rows: tuple[int, ...] = ()
    note: str = ""


def cut(tokens: str, small: str = "") -> str:
    """แสดงผลตัดคำ เช่น cut("ร้าน|อาหาร") -> "ร้าน | อาหาร" และต่อท้ายด้วย (small) ถ้ามี เช่น ความน่าจะเป็น"""
    text = " | ".join(tokens.split("|"))
    return f"{text} ({small})" if small else text


def html_table(table: Table) -> str:
    """สร้างตาราง HTML ล้วนที่ไม่มี class เพื่อให้วางที่อื่นแล้วยังเป็นตาราง

    คอลัมน์ตัวเลขชิดขวาด้วย style ในแต่ละช่อง (โปรแกรมส่วนใหญ่เก็บ style แบบนี้ไว้เมื่อวาง)
    """
    right = [' style="text-align:right"' if index in table.numeric else "" for index in range(len(table.head))]
    head = "".join(f"<th{align}>{html.escape(text)}</th>" for text, align in zip(table.head, right))
    rows = []
    for index, row in enumerate(table.rows):
        cells = [html.escape(text) for text in row]
        if index in table.bold_rows:
            cells = [f"<strong>{text}</strong>" for text in cells]
        rows.append("<tr>" + "".join(f"<td{align}>{text}</td>" for text, align in zip(cells, right)) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def markdown_table(table: Table) -> str:
    """สร้างตารางแบบ Markdown (GitHub Flavored) ขีดตั้งในเนื้อหาถูก escape เพื่อไม่ให้ถูกอ่านเป็นเส้นแบ่งช่อง"""
    def cell(text: str) -> str:
        """escape ขีดตั้งในหนึ่งช่อง"""
        return text.replace("|", "\\|")

    lines = ["| " + " | ".join(cell(text) for text in table.head) + " |",
             "|" + "|".join("---:" if index in table.numeric else "---" for index in range(len(table.head))) + "|"]
    for index, row in enumerate(table.rows):
        cells = [cell(text) for text in row]
        if index in table.bold_rows:
            cells = [f"**{text}**" for text in cells]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def results_page(tables: list[Table]) -> str:
    """ประกอบหน้า results.html: คำอธิบายวิธีคัดลอก แล้วตามด้วยแต่ละตารางพร้อมปุ่มและ Markdown ที่ซ่อนไว้"""
    sections = []
    for table in tables:
        note = f'<p class="note">{html.escape(table.note)}</p>' if table.note else ""
        sections.append(
            f"<section><h2>{html.escape(table.title)}</h2>{html_table(table)}{note}"
            '<div class="bar"><button type="button" data-copy="table">คัดลอกตาราง</button>'
            '<button type="button" data-copy="md">คัดลอกเป็น Markdown</button><span class="done"></span></div>'
            f"<textarea hidden>{html.escape(markdown_table(table), quote=False)}</textarea></section>")
    return (
        '<!doctype html><html lang="th"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>ผลการทดลอง LATTE WordSeg Lite</title><style>{PAGE_CSS}</style></head><body>"
        "<h1>ผลการทดลอง: LATTE WordSeg Lite</h1>"
        '<p class="sub">วัดบน AMD Ryzen 7 5700G (ใช้ 1 thread) และ NVIDIA GeForce RTX 3080 ด้วย iapp-ai 2.0.0 '
        "เมื่อ 23 พฤศจิกายน 2026</p>"
        '<div class="howto"><strong>วิธีคัดลอก</strong><br>'
        "กด “คัดลอกตาราง” แล้ววางในเว็บหรือเอกสารที่รองรับตาราง เช่น Google Docs หรือ Notion ตารางจะยังเป็นตาราง<br>"
        "ถ้าเว็บนั้นเขียนด้วย Markdown ให้กด “คัดลอกเป็น Markdown”<br>"
        "Medium ไม่มีตารางในตัว ตารางที่วางลงไปจะกลายเป็นข้อความทีละบรรทัด</div>"
        f"{''.join(sections)}<script>{PAGE_JS}</script></body></html>")


# ---------------------------------------------------------------- เนื้อหาของทุกตาราง เรียงตามลำดับในหน้า
PUBLISHED = "ผู้พัฒนา benchmark เผยแพร่"
TABLES = [
    # สเปกเครื่อง: จาก results/cpu.json (hardware) และรุ่นของ package ใน .venv
    Table("เครื่องที่ใช้ทดสอบ", ["รายการ", "รายละเอียด"], [
        ["CPU", "AMD Ryzen 7 5700G (8 cores / 16 threads) ทดสอบโดยใช้ 1 thread"],
        ["การ์ดจอ", "NVIDIA GeForce RTX 3080 12 GB"],
        ["RAM", "16 GB"],
        ["ระบบ", "Windows 11, Python 3.11.9"],
        ["โปรแกรม", "iapp-ai 2.0.0, ONNX Runtime 1.32.0"],
    ]),
    # คำอธิบาย: เขียนเอง ตามเอกสารของ package
    Table("ตัดคำ 3 ระดับ ต่างกันอย่างไร", ["ระดับ", "ตัดออกมาเป็น", "เหมาะกับงาน"], [
        ["long", "คำประสม เก็บคำที่รวมเป็นความหมายเดียวไว้ทั้งก้อน", "ดึง keyword, ชื่อเฉพาะ, ศัพท์เทคนิค"],
        ["short (ค่าเริ่มต้น)", "คำ แยกคำประสมออกเป็นคำ", "งานประมวลผลภาษาทั่วไป"],
        ["minimal", "หน่วยคำ แยกจนเหลือชิ้นเล็กสุดที่ยังมีความหมาย", "ระบบค้นหาที่อยากให้เจอครอบคลุม"],
    ]),
    # ตัวอย่างตัดคำและ n-best: จากผลที่ review.py พิมพ์
    Table("ตัวอย่าง: ร้านอาหารตามสั่งหน้าปากซอยอร่อยมาก", ["ระดับ", "ผลตัดคำ", "จำนวนชิ้น"], [
        ["long", cut("ร้านอาหาร|ตามสั่ง|หน้า|ปากซอย|อร่อย|มาก"), "6"],
        ["short", cut("ร้าน|อาหาร|ตามสั่ง|หน้า|ปาก|ซอย|อร่อย|มาก"), "8"],
        ["minimal", cut("ร้าน|อาหาร|ตาม|สั่ง|หน้า|ปาก|ซอย|อร่อย|มาก"), "9"],
    ], numeric=(2,)),
    Table("ตัวอย่าง: เจ้าหน้าที่การศึกษาเปิดเผยว่า", ["ระดับ", "ผลตัดคำ", "จำนวนชิ้น"], [
        ["long", cut("เจ้าหน้าที่|การศึกษา|เปิดเผย|ว่า"), "4"],
        ["short", cut("เจ้า|หน้าที่|การ|ศึกษา|เปิด|เผย|ว่า"), "7"],
        ["minimal", cut("เจ้า|หน้า|ที่|การ|ศึกษา|เปิด|เผย|ว่า"), "8"],
    ], numeric=(2,)),
    Table("n-best: ขอดูได้หลายคำตอบ เรียงตามความน่าจะเป็น", ["ข้อความ", "อันดับ 1", "อันดับ 2", "อันดับ 3"], [
        ["ตากลม", cut("ตา|กลม", "41.8%"), cut("ตากลม", "41.4%"), cut("ตาก|ลม", "8.3%")],
        ["ไปหามเหสี", cut("ไป|หา|มเหสี", "83.0%"), cut("ไป|หาม|เหสี", "0.5%"), cut("ไปหา|มเหสี", "0.2%")],
        ["แปลกว่า", cut("แปล|ก|ว่า", "53.0%"), cut("แปลก|ว่า", "23.3%"), cut("แปล|กว่า", "16.4%")],
        ["รอบอก", cut("รอ|บ|อก", "35.1%"), cut("รอ|บอก", "25.8%"), cut("รอบ|อก", "22.4%")],
    ], note="สองแถวล่าง คำตอบอันดับ 1 ไม่ใช่คำตอบที่ถูก ต้องดูอันดับถัดไป"),
    # ความแม่นของ LATTE: จาก results/f1.json (run_benchmark.py)
    Table("ความแม่น (word F1 เต็ม 100) แยกตามชุดทดสอบ",
          ["ชุดทดสอบ", "long", "short", "minimal", "short ที่ผู้พัฒนาเผยแพร่"], [
        ["TUD", "83.69", "90.23", "85.36", "90.27"],
        ["PUD", "86.20", "84.64", "80.35", "84.66"],
        ["Wisesight-160", "85.69", "88.53", "84.04", "88.32"],
        ["VISTEC", "92.73", "87.90", "76.47", "87.94"],
        ["เฉลี่ย 4 ชุด", "87.08", "87.82", "81.56", "87.80"],
        ["TALPCo (ชุดที่เราหามาเพิ่ม)", "88.69", "92.96", "92.15", "ไม่มี"],
    ], numeric=(1, 2, 3, 4), bold_rows=(4,)),
    # แถวแรกจาก results/f1.json ที่เหลือคือค่า avg4_commercial ที่ run_benchmark.py พิมพ์จากไฟล์ของผู้พัฒนา
    Table("เทียบกับตัวตัดคำอื่น: word F1 เฉลี่ย 4 ชุดเดียวกัน", ["ระบบ", "word F1", "ที่มาของตัวเลข"], [
        ["LATTE WordSeg Lite (short)", "87.82", "เราวัดเองบนเครื่องเรา"],
        ["SEFR CUT (ws1000)", "83.78", PUBLISHED],
        ["SEFR CUT (best)", "83.62", PUBLISHED],
        ["DeepCut", "83.59", PUBLISHED],
        ["OSKut (scads)", "83.50", PUBLISHED],
        ["AttaCut-SC", "83.12", PUBLISHED],
        ["PyThaiNLP newmm", "71.96", PUBLISHED],
    ], numeric=(1,), bold_rows=(0,)),
    # LATTE จาก results/f1.json (extra.talpco_tha) และ DeepCut จาก results/f1_deepcut.json (score_deepcut.py)
    Table("ชุด TALPCo: LATTE เทียบกับ DeepCut", ["ระบบ", "word F1", "boundary F1"], [
        ["LATTE ระดับ short", "92.96", "96.99"],
        ["LATTE ระดับ minimal", "92.15", "96.66"],
        ["DeepCut", "91.80", "95.97"],
        ["LATTE ระดับ long", "88.69", "95.09"],
    ], numeric=(1, 2), bold_rows=(0,),
        note="เฉลยของ TALPCo เริ่มจากผลของ DeepCut แล้วมีคนตรวจแก้ DeepCut จึงได้เปรียบบนชุดนี้"),
    # จาก results/f1_deepcut.json
    Table("DeepCut ที่เราติดตั้ง ได้คะแนนตรงกับที่ผู้พัฒนาเผยแพร่ (word F1)",
          ["ชุดทดสอบ", "เราวัดได้", "ผู้พัฒนาเผยแพร่"], [
        ["TUD", "85.50", "85.50"],
        ["PUD", "85.62", "85.62"],
        ["Wisesight-160", "83.13", "83.13"],
    ], numeric=(1, 2)),
    # DeepCut จาก results/deepcut.json และ LATTE จาก results/cpu_run2.json ซึ่งวัดต่อจาก DeepCut ทันที
    Table("ความเร็ว: ตัดข้อความ 1 ล้านตัวอักษร บน CPU 1 thread",
          ["ระบบ", "เวลาที่ใช้", "ตัวอักษรต่อวินาที", "เทียบกับ DeepCut"], [
        ["DeepCut (ทีละบรรทัด)", "379 วินาที", "2,636", "1 เท่า"],
        ["LATTE (ทีละบรรทัด)", "19.5 วินาที", "51,213", "เร็วกว่า 19.4 เท่า"],
        ["LATTE (ส่งเป็น batch)", "16.9 วินาที", "59,029", "เร็วกว่า 22.4 เท่า"],
    ], numeric=(1, 2, 3), bold_rows=(1,)),
    # จาก results/cpu.json, gpu.json และ dml.json ตรงกับที่ bench.py --report พิมพ์
    Table("ลองรันบนการ์ดจอ RTX 3080 เทียบกับ CPU (เวลาต่อ 1 ล้านตัวอักษร)",
          ["วิธีเรียก", "CPU 1 thread", "การ์ดจอ (CUDA)", "การ์ดจอ (DirectML)"], [
        ["ทีละบรรทัด", "19.9 วินาที", "34.9 วินาที (ช้ากว่า 1.75 เท่า)", "41.3 วินาที (ช้ากว่า 2.07 เท่า)"],
        ["batch 4,096 ตัวอักษร", "17.3 วินาที", "16.4 วินาที (เร็วกว่า 1.05 เท่า)", "18.8 วินาที (ช้ากว่า 1.09 เท่า)"],
        ["batch 32,768 ตัวอักษร", "17.9 วินาที", "14.7 วินาที (เร็วกว่า 1.22 เท่า)", "18.1 วินาที (ช้ากว่า 1.01 เท่า)"],
        ["batch 131,072 ตัวอักษร", "18.5 วินาที", "15.0 วินาที (เร็วกว่า 1.23 เท่า)", "17.1 วินาที (เร็วกว่า 1.08 เท่า)"],
    ], numeric=(1, 2, 3),
        note="วัดตอนเครื่องว่าง ในวงเล็บคือเทียบกับ CPU ผลฝั่งการ์ดจอแกว่งระหว่างรอบได้ราว 15%"),
    # จาก node_placement ใน results/gpu.json และ dml.json
    Table("เมื่อใช้การ์ดจอ งานของ model ไปรันที่ไหน (เวลาคำนวณของตัวอย่าง 200 บรรทัด)",
          ["ทางที่ใช้", "รันบนการ์ดจอ", "ค้างบน CPU", "คัดลอกข้อมูลข้ามฝั่งต่อรอบ"], [
        ["CUDA", "97 ms (13%)", "657 ms (87%)", "8 ครั้ง"],
        ["DirectML", "228 ms (24%)", "728 ms (76%)", "5 ครั้ง"],
    ], numeric=(1, 2, 3),
        note="ส่วนที่ค้างบน CPU คือชั้น LSTM แบบ INT8 ซึ่งไม่มีตัวรันบนการ์ดจอทั้งสองทาง (ทาง CUDA มีส่วน INT8 อื่นค้างด้วย)"),
    # คำนวณจากเวลาแบบทีละบรรทัดใน results/cpu.json, gpu.json และ dml.json หารด้วย 3,591 บรรทัด
    Table("ต้นทุนของการเรียกทีละบรรทัด (3,591 ครั้ง)", ["รันบน", "เวลารวม", "เฉลี่ยต่อครั้ง", "เพิ่มจาก CPU ต่อครั้ง"], [
        ["CPU 1 thread", "19.9 วินาที", "5.6 ms", "ไม่มี"],
        ["การ์ดจอ (CUDA)", "34.9 วินาที", "9.7 ms", "4.2 ms"],
        ["การ์ดจอ (DirectML)", "41.3 วินาที", "11.5 ms", "6.0 ms"],
    ], numeric=(1, 2, 3)),
    # จาก results/cpu_run1.json, cpu_run2.json, cpu.json, cpu_ort1244.json, gpu_run1.json, gpu.json,
    # dml_run1.json และ dml.json ตามลำดับแถว
    Table("ผลจับเวลาทุกรอบ (วินาทีต่อ 1 ล้านตัวอักษร แต่ละค่าคือค่าดีสุดจาก 3 ครั้ง)",
          ["รอบ", "ทีละบรรทัด", "batch 4,096", "batch 32,768", "batch 131,072", "สภาพเครื่องตอนวัด"], [
        ["CPU รอบ 1", "20.8", "17.0", "17.7", "18.1", "มีการติดตั้งโปรแกรมเดินอยู่"],
        ["CPU รอบ 2", "19.5", "16.9", "17.5", "17.9", "มีการดาวน์โหลดเดินอยู่"],
        ["CPU รอบ 3", "19.9", "17.3", "17.9", "18.5", "เครื่องว่าง"],
        ["CPU รอบ 4 (ONNX Runtime 1.24.4)", "20.2", "17.4", "17.6", "17.9", "เครื่องว่าง"],
        ["CUDA รอบ 1", "40.0", "16.0", "16.5", "15.5", "เครื่องว่าง"],
        ["CUDA รอบ 2", "34.9", "16.4", "14.7", "15.0", "เครื่องว่าง"],
        ["DirectML รอบ 1", "46.3", "19.5", "18.6", "17.8", "มีการดาวน์โหลดเดินอยู่"],
        ["DirectML รอบ 2", "41.3", "18.8", "18.1", "17.1", "เครื่องว่าง"],
    ], numeric=(1, 2, 3, 4),
        note="CPU รอบ 1 ถึง 3 ใช้ ONNX Runtime 1.32 ส่วนการ์ดจอทุกรอบใช้ 1.24.4 "
             "ตารางเทียบกับ DeepCut ใช้ CPU รอบ 2 ซึ่งวัดต่อจาก DeepCut ทันที "
             "ตารางการ์ดจอใช้ CPU รอบ 3, CUDA รอบ 2 และ DirectML รอบ 2"),
    # สองแถวกลางจาก bench.py --report แถวแรกเทียบ cpu.tokens.txt กับ cpu_ort1244.tokens.txt
    # แถวสุดท้ายจาก results/claims.json (check_claims.py)
    Table("ผลตัดคำเหมือนกันแค่ไหน (จาก 3,591 บรรทัด ระดับ short)", ["เทียบอะไรกับอะไร", "บรรทัดที่เหมือนกัน", "สัดส่วน"], [
        ["CPU บน ONNX Runtime 1.32 กับ 1.24.4", "3,591", "100%"],
        ["CPU กับการ์ดจอ (DirectML)", "3,572", "99.5%"],
        ["CPU กับการ์ดจอ (CUDA)", "3,535", "98.4%"],
        ["CPU เรียกทีละบรรทัด กับ CPU ส่งเป็น batch", "3,486", "97.1%"],
    ], numeric=(1, 2)),
    # RAM จาก results/ram.json (check_ram.py) และขนาดไฟล์จาก results/claims.json (check_claims.py)
    Table("หน่วยความจำและขนาดไฟล์", ["รายการ", "ที่วัดได้"], [
        ["RAM หลังโหลด model", "79 MB"],
        ["RAM สูงสุดเมื่อเรียกทีละบรรทัด", "97 MB"],
        ["RAM สูงสุดเมื่อส่ง 1 ล้านตัวอักษรใน batch เดียว", "216 MB"],
        ["ขนาด model กับ dictionary บนดิสก์", "3.82 MB"],
    ], numeric=(1,)),
    # ข้อสรุป: เขียนเอง จากผลในตารางข้างบนเทียบกับประกาศของผู้พัฒนา
    Table("สิ่งที่ผู้พัฒนาบอก กับสิ่งที่เราวัดได้", ["สิ่งที่ผู้พัฒนาบอก", "ผลบนเครื่องเรา", "สรุป"], [
        ["ตัดได้ 3 ระดับ และขอดูหลายคำตอบได้", "ใช้ได้จริง ผลตรงกับตัวอย่างในประกาศ", "ตรง"],
        ["แม่นที่สุดโดยรวม (85.6 จาก 7 ชุด)", "ตรวจได้ 4 จาก 7 ชุด ได้ 87.82 สูงสุดในกลุ่ม", "ตรง (ตรวจได้ 4 ชุด)"],
        ["เร็วกว่า DeepCut 23 เท่า", "เร็วกว่า 19.4 เท่า", "ใกล้เคียง"],
        ["1 ล้านตัวอักษรราว 17 วินาที", "19.5 วินาที", "ใกล้เคียง"],
        ["ส่งเป็น batch ราว 10 วินาที", "16.9 วินาที", "ช้ากว่าที่บอก"],
        ["model กับ dictionary 3.8 MB", "3.82 MB", "ตรง"],
        ["ใช้ RAM 136 MB", "97 MB เมื่อเรียกทีละบรรทัด และ 216 MB เมื่อส่ง 1 ล้านตัวอักษรรวดเดียว", "วัดคนละเงื่อนไข"],
        ["รัน offline ได้ ติดตั้งคำสั่งเดียว", "ตัดคำได้ปกติเมื่อปิดเน็ต", "ตรง"],
        ["ใช้เชิงพาณิชย์ได้ (CC BY-SA 4.0)", "ตรงกับเอกสาร licence ที่มากับโปรแกรม", "ตรง"],
    ]),
    # จำนวนบรรทัดและคำ: นับจากไฟล์ชุดทดสอบที่ prepare_data.py เตรียม
    Table("ข้อมูลที่ใช้ทดสอบ (ทุกชุดมีคนตัดคำเป็นเฉลย และแจกต่อได้)", ["ชุด", "เนื้อหา", "Licence", "บรรทัด", "คำในเฉลย"], [
        ["TUD", "UD Thai-TUD ส่วนทดสอบ", "CC BY-SA 4.0", "363", "7,683"],
        ["PUD", "UD Thai-PUD", "CC BY-SA 3.0", "1,000", "22,330"],
        ["Wisesight-160", "ข้อความโซเชียล", "CC0", "160", "3,374"],
        ["VISTEC", "ข้อความโซเชียล", "CC BY-SA 3.0", "10,000", "677,448"],
        ["TALPCo", "ประโยคสั้นแปลจากภาษาญี่ปุ่น", "CC BY 4.0", "1,368", "10,923"],
    ], numeric=(3, 4), note="4 ชุดแรกมาจาก benchmark ของผู้พัฒนา ชุดสุดท้ายเราหามาเพิ่มเอง"),
]


def main() -> None:
    """ตรวจว่าทุกแถวมีจำนวนช่องครบ แล้วเขียนหน้า results.html"""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")            # ให้ print ภาษาไทยได้บน Windows
    for table in TABLES:
        for row in table.rows:
            if len(row) != len(table.head):                 # กันพิมพ์ตกช่อง ซึ่งจะทำให้ตารางเลื่อนคอลัมน์
                raise SystemExit(f'ตาราง "{table.title}" มีแถวที่จำนวนช่องไม่เท่าหัวตาราง: {row}')
    PAGE.write_text(results_page(TABLES), encoding="utf-8", newline="\n")
    print(f"{PAGE.relative_to(HERE.parent)}  {len(TABLES)} ตาราง")


if __name__ == "__main__":
    main()
