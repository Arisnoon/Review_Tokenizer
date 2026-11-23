# Review_Tokenizer

รีวิวและทดสอบ **OpenThai 2.0 LATTE WordSeg Lite** ตัวตัดคำภาษาไทยของทีม OpenThai และ iApp
โดยติดตั้งและวัดเองบนเครื่องของเรา แล้วเทียบกับสิ่งที่ประกาศของผู้พัฒนาบอกไว้ทีละข้อ

ทุกตัวเลขใน repo นี้วัดเมื่อ 23 พฤศจิกายน 2026 ด้วย `iapp-ai` 2.0.0 และสร้างซ้ำได้ด้วยสคริปต์ที่อยู่ที่นี่

**บทความรีวิวฉบับเต็ม:** [รีวิวตัวตัดคำภาษาไทย LATTE WordSeg Lite บน Medium](https://medium.com/@arisa.ump/%E0%B8%A3%E0%B8%B5%E0%B8%A7%E0%B8%B4%E0%B8%A7%E0%B8%95%E0%B8%B1%E0%B8%A7%E0%B8%95%E0%B8%B1%E0%B8%94%E0%B8%84%E0%B8%B3%E0%B8%A0%E0%B8%B2%E0%B8%A9%E0%B8%B2%E0%B9%84%E0%B8%97%E0%B8%A2-latte-wordseg-lite-%E0%B8%81%E0%B8%B1%E0%B8%99-%E0%B9%81%E0%B8%A5%E0%B9%89%E0%B8%A7%E0%B8%A1%E0%B8%B2%E0%B8%94%E0%B8%B9%E0%B8%95%E0%B9%88%E0%B8%AD%E0%B8%81%E0%B8%B1%E0%B8%99%E0%B8%A7%E0%B9%88%E0%B8%B2-%E0%B8%9E%E0%B8%AD%E0%B9%80%E0%B8%9B%E0%B8%A5%E0%B8%B5%E0%B9%88%E0%B8%A2%E0%B8%99-hardware-batch-size-98fea4ad8b95)

## สารบัญ

- [สิ่งที่ทดสอบ](#สิ่งที่ทดสอบ)
- [ผลการทดสอบ](#ผลการทดสอบ)
- [ตัดคำ 3 ระดับคืออะไร](#ตัดคำ-3-ระดับคืออะไร)
- [การติดตั้ง](#การติดตั้ง)
- [ลำดับการรัน](#ลำดับการรัน)
- [โครงสร้างไฟล์](#โครงสร้างไฟล์)
- [ข้อมูลที่ใช้และ licence](#ข้อมูลที่ใช้และ-licence)
- [วิธีวัดและข้อควรระวัง](#วิธีวัดและข้อควรระวัง)
- [สิ่งที่ไม่ได้ตรวจ](#สิ่งที่ไม่ได้ตรวจ)
- [เครดิตและลิงก์](#เครดิตและลิงก์)

## สิ่งที่ทดสอบ

ประกาศของผู้พัฒนาบอกว่า model นี้ตัดคำได้ 3 ระดับ ขอดูได้หลายคำตอบ แม่นที่สุดโดยรวม เร็วกว่า DeepCut 23 เท่า
เล็กแค่ 3.8 MB รัน offline ได้ และใช้เชิงพาณิชย์ได้ repo นี้ตรวจแต่ละข้อดังนี้

| เรื่อง | ตรวจอย่างไร | สคริปต์ |
|---|---|---|
| ตัดได้ 3 ระดับ และหลายคำตอบ (n-best) | รันกับประโยคตัวอย่างแล้วดูผล | `review.py` |
| ความแม่น | วัด word F1 บนชุดทดสอบที่มีเฉลยจากคน เทียบกับค่าที่ผู้พัฒนาเผยแพร่ | `run_benchmark.py`, `score_deepcut.py` |
| ความเร็ว | จับเวลาตัดข้อความ 1 ล้านตัวอักษร เทียบกับ DeepCut บนเครื่องเดียวกัน | `bench.py`, `bench_deepcut.py` |
| ใช้การ์ดจอแล้วเร็วขึ้นไหม | รัน model เดียวกันบน RTX 3080 ผ่าน CUDA และ DirectML | `bench.py` |
| ขนาดไฟล์, RAM, รัน offline | วัดจาก process จริง และปิดเครือข่ายแล้วลองรัน | `check_ram.py`, `check_claims.py` |

## ผลการทดสอบ

ผลทั้งหมด 18 ตารางอยู่ที่ [blog/results.html](blog/results.html) ซึ่งคัดลอกไปวางในเว็บหรือเอกสารได้
ตารางหลักมีดังนี้

### เครื่องที่ใช้ทดสอบ

| รายการ | รายละเอียด |
|---|---|
| CPU | AMD Ryzen 7 5700G (8 cores / 16 threads) ทดสอบโดยใช้ 1 thread |
| การ์ดจอ | NVIDIA GeForce RTX 3080 12 GB |
| RAM | 16 GB |
| ระบบ | Windows 11, Python 3.11.9 |
| โปรแกรม | iapp-ai 2.0.0, ONNX Runtime 1.32.0 |

### 01 · Accuracy: ความแม่นยำของ LATTE WordSeg Lite

Word F1 (คะแนนเต็ม 100) เปรียบเทียบการตัดคำ 3 ระดับ

| ชุดทดสอบ | long | short | minimal | ผู้พัฒนา (short) |
|---|---:|---:|---:|---:|
| TUD | 83.69 | 90.23 | 85.36 | 90.27 |
| PUD | 86.20 | 84.64 | 80.35 | 84.66 |
| Wisesight-160 | 85.69 | 88.53 | 84.04 | 88.32 |
| VISTEC | 92.73 | 87.90 | 76.47 | 87.94 |
| **เฉลี่ย 4 ชุด** | **87.08** | **87.82** | **81.56** | **87.80** |
| TALPCo (เพิ่มเอง) | 88.69 | 92.96 | 92.15 | — |

\* TALPCo รายงานแยกต่างหาก ไม่รวมในค่าเฉลี่ย Benchmark ต้นฉบับ 4 ชุด

### 02 · CPU Speed: LATTE เร็วกว่า DeepCut แค่ไหน?

ทดสอบข้อความ 1,000,019 ตัวอักษร บน AMD Ryzen 7 5700G โดยใช้ CPU 1 thread

| ระบบ | เวลาที่ใช้ | ตัวอักษร/วินาที | เทียบกับ DeepCut |
|---|---:|---:|---:|
| DeepCut (ทีละบรรทัด) | 379 วินาที | 2,636 | 1 เท่า |
| **LATTE (ทีละบรรทัด)** | **19.5 วินาที** | **51,213** | **เร็วกว่า 19.4 เท่า** |
| LATTE (ส่งเป็น Batch) | 16.9 วินาที | 59,029 | เร็วกว่า 22.4 เท่า |

\* อัตราส่วนแบบ Batch เปรียบเทียบ LATTE Batch กับ DeepCut ที่เรียกทีละบรรทัด

### 03 · CPU vs GPU: CPU เทียบกับ RTX 3080

เวลาในการตัดคำ 1,000,019 ตัวอักษร วัดขณะเครื่องว่าง

| วิธีเรียก | CPU (1 Thread) | RTX 3080 (CUDA) | RTX 3080 (DirectML) |
|---|---:|---:|---:|
| ทีละบรรทัด | 19.9 วินาที | 34.9 วินาที<br>ช้ากว่า 1.75 เท่า | 41.3 วินาที<br>ช้ากว่า 2.07 เท่า |
| Batch 4,096 | 17.3 วินาที | 16.4 วินาที<br>เร็วกว่า 1.05 เท่า | 18.8 วินาที<br>ช้ากว่า 1.09 เท่า |
| Batch 32,768 | 17.9 วินาที | 14.7 วินาที<br>เร็วกว่า 1.22 เท่า | 18.1 วินาที<br>ช้ากว่า 1.01 เท่า |
| Batch 131,072 | 18.5 วินาที | 15.0 วินาที<br>เร็วกว่า 1.23 เท่า | 17.1 วินาที<br>เร็วกว่า 1.08 เท่า |

\* ค่าที่แสดงเป็นเวลาที่ดีที่สุดจากการรัน 3 ครั้งต่อรอบวัด

\* ความต่างข้ามรอบ: CPU ราว 3–7% และ GPU ราว 15%

### 04 · Claim Check: ผู้พัฒนาบอกอะไร แล้วเราวัดได้เท่าไร?

เทียบคำกล่าวจากประกาศกับผลทดสอบบนเครื่องของเรา

| สิ่งที่ผู้พัฒนาระบุ | ผลบนเครื่องเรา | ข้อสรุป |
|---|---|---|
| ตัดคำได้ 3 ระดับ และแสดงหลายคำตอบ (N-best) | ใช้งานได้จริง ผลตรงกับตัวอย่างในประกาศ | ตรง |
| แม่นที่สุดโดยรวม 85.6 คะแนน จาก 7 ชุด | ทดสอบได้ 4 จาก 7 ชุด short เฉลี่ย 87.82 (ผู้พัฒนา 87.80 บน 4 ชุด) | ยืนยันได้เฉพาะ 4 ชุด |
| เร็วกว่า DeepCut 23 เท่า | LATTE ทีละบรรทัด: 19.4 เท่า<br>LATTE Batch: 22.4 เท่า | ใกล้เคียง |
| 1 ล้านตัวอักษร ใช้เวลาราว 17 วินาที | LATTE ทีละบรรทัด 19.5 วินาที | ใกล้เคียง |
| ส่งเป็น Batch ใช้เวลาราว 10 วินาที | LATTE Batch 16.9 วินาที | ช้ากว่าที่ระบุ |
| โมเดล + Dictionary ขนาด 3.8 MB | วัดได้ 3.82 MB | ตรง |
| ใช้ RAM 136 MB | ทีละบรรทัด 97 MB<br>ส่งข้อความ 1 ล้านตัวอักษรรวดเดียว 216 MB | ต่างเงื่อนไขการวัด |
| รัน Offline ได้ ติดตั้งคำสั่งเดียว | ปิดอินเทอร์เน็ตแล้วใช้งานได้ตามปกติ | ตรง |
| ใช้เชิงพาณิชย์ได้ (CC BY-SA 4.0) | สอดคล้องกับเอกสาร Licence ที่มากับโปรแกรม | ตรง |

\* “ยืนยันได้เฉพาะ 4 ชุด” ไม่ใช่การยืนยันอันดับรวมจาก Benchmark ครบ 7 ชุด

\* ความเร็วบนเครื่องต่างรุ่นอาจไม่เท่ากัน; RAM ขึ้นอยู่กับวิธีส่งข้อมูล

## ตัดคำ 3 ระดับคืออะไร

ภาษาไทยเขียนติดกันโดยไม่เว้นวรรคระหว่างคำ และ "คำ" ไม่มีนิยามเดียว model นี้จึงให้เลือกความละเอียดของการตัดได้
3 ระดับจาก model ตัวเดียว ด้วยตัวเลือก `unit`

| ระดับ | ตัดออกมาเป็น | เหมาะกับงาน | ตัวอย่าง: ร้านอาหารตามสั่งหน้าปากซอยอร่อยมาก |
|---|---|---|---|
| `long` | คำประสม | ดึง keyword, ชื่อเฉพาะ, ศัพท์เทคนิค | ร้านอาหาร / ตามสั่ง / หน้า / ปากซอย / อร่อย / มาก |
| `short` (ค่าเริ่มต้น) | คำ | งานประมวลผลภาษาทั่วไป | ร้าน / อาหาร / ตามสั่ง / หน้า / ปาก / ซอย / อร่อย / มาก |
| `minimal` | หน่วยคำ | ระบบค้นหาที่อยากให้เจอครอบคลุม | ร้าน / อาหาร / ตาม / สั่ง / หน้า / ปาก / ซอย / อร่อย / มาก |

```python
from iapp_ai.nlp import word_tokenize, word_tokenize_nbest

text = "ร้านอาหารตามสั่งหน้าปากซอยอร่อยมาก"
word_tokenize(text)                    # ระดับ short (ค่าเริ่มต้น)
word_tokenize(text, unit="long")       # หรือ unit="minimal"
word_tokenize_nbest("ตากลม", n=3)      # ขอดู 3 คำตอบ เรียงตามความน่าจะเป็น
```

## การติดตั้ง

### สิ่งที่ต้องมี

- Windows (ทดสอบบน Windows 11 เท่านั้น ยังไม่ได้ลองบนระบบอื่น) ส่วนที่เขียนไว้เฉพาะ Windows คือการอ่านชื่อ CPU ค่า RAM สูงสุด และการรันบนการ์ดจอ
- Python 3.11 (`iapp-ai` 2.0.0 ต้องการ Python 3.10 ขึ้นไป)
- [uv](https://docs.astral.sh/uv/) สำหรับสร้าง environment (ใช้ `python -m venv` กับ `pip` แทนได้)
- การ์ดจอ NVIDIA เฉพาะเมื่อต้องการทำการทดลองฝั่งการ์ดจอ

### โหลด repo

```powershell
git clone https://github.com/Arisnoon/Review_Tokenizer.git
cd Review_Tokenizer
```

### สร้าง environment

การทดลองใช้ environment แยกกัน 4 ตัว เพราะแต่ละตัวต้องการ ONNX Runtime คนละแบบ ซึ่งติดตั้งรวมกันไม่ได้
ถ้าต้องการแค่ความแม่นและความเร็วบน CPU สร้างเฉพาะตัวแรกก็พอ

| environment | ใช้กับ | package หลัก |
|---|---|---|
| `.venv` | สคริปต์ทั้งหมดที่รันบน CPU | iapp-ai 2.0.0, onnxruntime 1.32.0, numpy 2.4.6, psutil |
| `.venv-deepcut` | `bench_deepcut.py`, `score_deepcut.py` | deepcut 0.7.0.0, tensorflow 2.15.1 |
| `.venv-gpu12` | `bench.py --device gpu` | iapp-ai 2.0.0, onnxruntime-gpu 1.24.4, ชุด CUDA 12 และ cuDNN 9 |
| `.venv-dml` | `bench.py --device dml` | iapp-ai 2.0.0, onnxruntime-directml 1.24.4 |

```powershell
# 1) environment หลัก (จำเป็น)
uv venv .venv --python 3.11
uv pip install --python .venv "iapp-ai[nlp]==2.0.0" psutil

# 2) DeepCut สำหรับใช้เป็นตัวเทียบ (ไม่บังคับ โหลด TensorFlow ราว 300 MB)
uv venv .venv-deepcut --python 3.11
uv pip install --python .venv-deepcut "deepcut==0.7.0.0" "tensorflow==2.15.*"

# 3) การ์ดจอผ่าน CUDA 12 (ไม่บังคับ โหลดราว 1.8 GB)
uv venv .venv-gpu12 --python 3.11
uv pip install --python .venv-gpu12 "iapp-ai==2.0.0" numpy "onnxruntime-gpu[cuda,cudnn]==1.24.4"

# 4) การ์ดจอผ่าน DirectML (ไม่บังคับ)
uv venv .venv-dml --python 3.11
uv pip install --python .venv-dml "iapp-ai==2.0.0" numpy "onnxruntime-directml==1.24.4"
```

ข้อควรรู้ของ environment ฝั่งการ์ดจอ

- ติดตั้ง `iapp-ai` แบบไม่มี `[nlp]` เพราะ `[nlp]` จะดึง `onnxruntime` รุ่น CPU มาด้วย ซึ่งชนกับรุ่นการ์ดจอ
- driver ของการ์ดจอต้องรองรับ CUDA 12 เครื่องที่ใช้ทดสอบมี driver 560.94 ซึ่งรองรับถึง CUDA 12.6
  จึงใช้ `onnxruntime-gpu` 1.24.4 ส่วนรุ่น 1.32.0 ใช้ CUDA 13 และรันบน driver นี้ไม่ได้ (ขึ้น `CUDA failure 801`)
- DirectML ใช้ได้กับการ์ดจอที่รองรับ DirectX 12 และไม่ต้องติดตั้ง CUDA

### เตรียมข้อมูล

```powershell
.venv\Scripts\python.exe -X utf8 prepare_data.py
```

คำสั่งนี้โหลดชุดทดสอบและตัวคิดคะแนนของ benchmark (ราว 16 MB) กับ TALPCo ภาษาไทย ตรวจ SHA-256 ทุกไฟล์
แล้วสร้างข้อความจับเวลา 1 ล้านตัวอักษร ได้โฟลเดอร์ `wordseg-bench/` และ `data/`

## ลำดับการรัน

รายละเอียดของแต่ละขั้น ไฟล์ที่อ่านและเขียน และผลที่ควรได้ อยู่ใน [docs/WORKFLOW.md](docs/WORKFLOW.md)

| ขั้น | ทำอะไร | คำสั่ง | environment | เวลาโดยประมาณ |
|---:|---|---|---|---|
| 1 | เตรียมข้อมูล | `prepare_data.py` | `.venv` | ขึ้นกับความเร็วเน็ต |
| 2 | ดูผลตัดคำ 3 ระดับ และ n-best | `review.py` | `.venv` | ไม่กี่วินาที |
| 3 | ความแม่นของ LATTE | `run_benchmark.py` | `.venv` | 3.5 นาที |
| 4 | ความแม่นของ DeepCut | `score_deepcut.py` | `.venv-deepcut` | 2.5 นาที |
| 5 | ความเร็วของ LATTE บน CPU | `bench.py --device cpu` | `.venv` | 4 นาที |
| 6 | ความเร็วของ DeepCut | `bench_deepcut.py` | `.venv-deepcut` | 6.5 นาที |
| 7 | ความเร็วบนการ์ดจอผ่าน CUDA | `bench.py --device gpu` | `.venv-gpu12` | 4.5 นาที |
| 8 | ความเร็วบนการ์ดจอผ่าน DirectML | `bench.py --device dml` | `.venv-dml` | 5 นาที |
| 9 | เทียบ CPU กับการ์ดจอ | `bench.py --report` | `.venv` | ทันที |
| 10 | หน่วยความจำ | `check_ram.py` | `.venv` | 1 นาที |
| 11 | ขนาดไฟล์, รัน offline, ผล batch | `check_claims.py` | `.venv` | 1 นาที |
| 12 | สร้างหน้าตารางผล | `blog\make_tables.py` | ตัวไหนก็ได้ | ทันที |

ตัวอย่างการรันทั้งชุดบน CPU

```powershell
.venv\Scripts\python.exe -X utf8 prepare_data.py
.venv\Scripts\python.exe -X utf8 review.py
.venv\Scripts\python.exe -X utf8 run_benchmark.py
.venv\Scripts\python.exe -X utf8 bench.py --device cpu
.venv\Scripts\python.exe -X utf8 check_ram.py
.venv\Scripts\python.exe -X utf8 check_claims.py
```

ถ้าอยากลองเร็ว ๆ ก่อนรันจริง `bench.py` และ `bench_deepcut.py` มีตัวเลือก `--chars` ให้ใช้ข้อความแค่บางส่วน
และจะไม่บันทึกผล เช่น `bench.py --device cpu --chars 100000`

## โครงสร้างไฟล์

```text
Review_Tokenizer/
├── README.md
├── docs/
│   └── WORKFLOW.md        ขั้นตอนการทดลองโดยละเอียด
├── common.py              ของใช้ร่วมของทุกสคริปต์: ตำแหน่งไฟล์ รายชื่อชุดทดสอบ การอ่านเขียนไฟล์ ตัวคิดคะแนน
├── prepare_data.py        เตรียมข้อมูลทั้งหมด: ไฟล์ของ benchmark ข้อความจับเวลา และ TALPCo
├── review.py              ดูผลตัดคำ 3 ระดับ และหลายคำตอบ (n-best)
├── run_benchmark.py       วัดความแม่นของ LATTE
├── score_deepcut.py       วัดความแม่นของ DeepCut เพื่อใช้เป็นตัวเทียบ
├── bench.py               จับเวลา LATTE บน CPU, CUDA และ DirectML และพิมพ์ตารางเทียบ
├── bench_deepcut.py       จับเวลา DeepCut
├── check_ram.py           วัดหน่วยความจำ
├── check_claims.py        ตรวจขนาดไฟล์ การรัน offline และผล batch เทียบกับทีละบรรทัด
├── blog/
│   ├── make_tables.py     สร้างหน้าตารางผล
│   └── results.html       ผลการทดลองทั้งหมด 18 ตาราง พร้อมปุ่มคัดลอก
└── results/               ไฟล์ผลของการทดลอง (JSON)
```

โฟลเดอร์ที่ไม่ได้อยู่ใน repo เพราะสร้างใหม่ได้: `wordseg-bench/` และ `data/` (จาก `prepare_data.py`),
`.venv*/` (จากขั้นติดตั้ง) และ `results/*.tokens.txt` (จาก `bench.py`)

ไฟล์ใน `results/`

| ไฟล์ | มาจาก | เนื้อหา |
|---|---|---|
| `f1.json` | `run_benchmark.py` | word F1 และ boundary F1 ของ LATTE ทั้ง 3 ระดับ |
| `f1_deepcut.json` | `score_deepcut.py` | word F1 ของ DeepCut |
| `cpu.json`, `gpu.json`, `dml.json` | `bench.py` | เวลาของ LATTE บน CPU, CUDA และ DirectML (รอบล่าสุด) |
| `cpu_run1.json`, `cpu_run2.json`, `gpu_run1.json`, `dml_run1.json` | `bench.py` | ผลของรอบก่อนหน้าที่คัดลอกเก็บไว้ ใช้ดูความแกว่งระหว่างรอบ |
| `cpu_ort1244.json` | `bench.py --device cpu` ที่รันใน `.venv-gpu12` | เวลาบน CPU ด้วย ONNX Runtime 1.24.4 ซึ่งเป็นรุ่นเดียวกับที่การ์ดจอใช้ |
| `deepcut.json` | `bench_deepcut.py` | เวลาของ DeepCut |
| `ram.json` | `check_ram.py` | หน่วยความจำของ 3 แบบการใช้งาน |
| `claims.json` | `check_claims.py` | ขนาดไฟล์ ผลการรัน offline และจำนวนบรรทัดที่ batch ตัดต่างจากทีละบรรทัด |

## ข้อมูลที่ใช้และ licence

ใช้เฉพาะชุดทดสอบที่มีเฉลยจากคน และ licence ให้แจกต่อได้

| ชุด | เนื้อหา | ที่มา | Licence | บรรทัด | คำในเฉลย |
|---|---|---|---|---:|---:|
| TUD | UD Thai-TUD ส่วนทดสอบ | benchmark ของผู้พัฒนา | CC BY-SA 4.0 | 363 | 7,683 |
| PUD | UD Thai-PUD | benchmark ของผู้พัฒนา | CC BY-SA 3.0 | 1,000 | 22,330 |
| Wisesight-160 | ข้อความโซเชียล | benchmark ของผู้พัฒนา | CC0 | 160 | 3,374 |
| VISTEC | ข้อความโซเชียล | benchmark ของผู้พัฒนา | CC BY-SA 3.0 | 10,000 | 677,448 |
| TALPCo | ประโยคสั้นแปลจากภาษาญี่ปุ่น | หามาเพิ่มเอง | CC BY 4.0 | 1,368 | 10,923 |

- **ชุดที่ไม่ใช้:** benchmark ต้นฉบับมี 7 ชุด อีก 3 ชุดคือ BEST, ORCHID และชุดทดสอบของ khammoon เป็น licence
  แบบไม่ให้ใช้เชิงพาณิชย์ และ benchmark ไม่ได้แจกมาด้วย จึงไม่โหลดและไม่วัด
- **TALPCo:** เฉลยเริ่มจากผลของ DeepCut แล้วมีคนตรวจแก้ DeepCut จึงได้เปรียบบนชุดนี้
  ต้นฉบับมี 1,372 ประโยค ใช้ได้ 1,368 เพราะอีก 4 ประโยคมีคำที่ไม่ตรงกับประโยคดิบ
- **ข้อความจับเวลา:** 3,591 บรรทัดแรกของ VISTEC รวม 1,000,019 ตัวอักษร ใช้กับการจับเวลาทุกรายการ
- **การตรวจไฟล์:** ทุกไฟล์โหลดจาก revision ที่ระบุตายตัว และตรวจ SHA-256 ก่อนใช้

## วิธีวัดและข้อควรระวัง

- **ใช้ CPU 1 thread ทุกรายการ** ตามเงื่อนไขที่ผู้พัฒนาใช้ `iapp-ai` ตั้ง 1 thread เป็นค่าเริ่มต้นอยู่แล้ว ส่วน DeepCut ถูกบังคับให้ใช้ 1 thread ในสคริปต์
- **ความเร็วเก็บค่าที่ดีที่สุดจาก 3 ครั้ง** ในแต่ละรอบ และเวลาโหลด model ไม่นับรวม ยกเว้น DeepCut ซึ่งรันช้าจึงวัดรอบเดียว
- **ตัวเลขความเร็วแกว่งระหว่างรอบ** CPU ต่างกันราว 3–7% การ์ดจอต่างกันได้ราว 15% ผลต่างของ CUDA แบบ batch (3–23%) จึงควรอ่านว่า "เร็วขึ้นเล็กน้อย"
- **CPU เครื่องเราเป็นคนละรุ่นกับของผู้พัฒนา** เขาวัดบน Intel Xeon Platinum 8468 กับข้อความอีกชุดหนึ่ง ตัวเลขความเร็วจึงไม่จำเป็นต้องเท่ากัน
- **คะแนนความแม่นไม่แกว่ง** การเรียกทีละบรรทัดได้ผลตัดคำเหมือนเดิมทุกครั้ง
- **ผลแบบ batch ต่างจากทีละบรรทัดได้เล็กน้อย** พบ 105 จาก 3,591 บรรทัด (2.9%) ถ้าต้องการผลเหมือนเดิมทุกครั้งให้เรียกทีละบรรทัด
- **ผลบนการ์ดจอต่างจาก CPU เล็กน้อย** ตรงกัน 98.4% ของบรรทัดผ่าน CUDA และ 99.5% ผ่าน DirectML
- **การ์ดจอช่วยได้น้อย** เพราะชั้น LSTM แบบ INT8 ของ model ไม่มีตัวรันบนการ์ดจอ งานส่วนใหญ่ (76–87% ของเวลาคำนวณ) จึงค้างอยู่บน CPU
- **ระวัง ONNX Runtime ถอยไปใช้ CPU เงียบ ๆ** เมื่อการ์ดจอรันไม่สำเร็จ โปรแกรมจะรันต่อบน CPU โดยไม่หยุด `bench.py` จึงตรวจซ้ำหลังรันจริง และไม่บันทึกผลถ้าพบว่าถอยไป CPU
- **ONNX Runtime รุ่นเก่ากับ cuDNN รุ่นใหม่** รุ่น 1.24.4 ไม่โหลด `cudnn_engines_tensor_ir64_9.dll` ของ cuDNN 9.28 ทำให้รัน CUDA ไม่ได้ `bench.py` แก้โดยเพิ่มโฟลเดอร์ของ package `nvidia-*` เข้า path ก่อนสร้าง session

## สิ่งที่ไม่ได้ตรวจ

- **ค่าเฉลี่ย 85.6 จาก 7 ชุด** ตรวจได้เฉพาะ 4 ชุดที่แจกต่อได้ ตามเหตุผลในหัวข้อข้อมูล
- **คะแนนของตัวตัดคำอื่น** ใช้ค่าที่ผู้พัฒนา benchmark เผยแพร่ ยกเว้น DeepCut ซึ่งวัดซ้ำ 3 ชุดและได้ค่าตรงกัน
- **การเทรนบน GPU NVIDIA H100** เป็นข้อมูลจากเปเปอร์ของผู้พัฒนา ตรวจจากภายนอกไม่ได้

## เครดิตและลิงก์

- บทความของการทดลองนี้: [รีวิวตัวตัดคำภาษาไทย LATTE WordSeg Lite บน Medium](https://medium.com/@arisa.ump/%E0%B8%A3%E0%B8%B5%E0%B8%A7%E0%B8%B4%E0%B8%A7%E0%B8%95%E0%B8%B1%E0%B8%A7%E0%B8%95%E0%B8%B1%E0%B8%94%E0%B8%84%E0%B8%B3%E0%B8%A0%E0%B8%B2%E0%B8%A9%E0%B8%B2%E0%B9%84%E0%B8%97%E0%B8%A2-latte-wordseg-lite-%E0%B8%81%E0%B8%B1%E0%B8%99-%E0%B9%81%E0%B8%A5%E0%B9%89%E0%B8%A7%E0%B8%A1%E0%B8%B2%E0%B8%94%E0%B8%B9%E0%B8%95%E0%B9%88%E0%B8%AD%E0%B8%81%E0%B8%B1%E0%B8%99%E0%B8%A7%E0%B9%88%E0%B8%B2-%E0%B8%9E%E0%B8%AD%E0%B9%80%E0%B8%9B%E0%B8%A5%E0%B8%B5%E0%B9%88%E0%B8%A2%E0%B8%99-hardware-batch-size-98fea4ad8b95)
- Model: [OpenThai2.0-LATTE-WordSeg-Lite](https://huggingface.co/iapp/OpenThai2.0-LATTE-WordSeg-Lite) โดย OpenThai และ iApp Technology (model และ dictionary เป็น CC BY-SA 4.0)
- Benchmark: [OpenThai2.0-WordSeg-Benchmark](https://huggingface.co/datasets/iapp/OpenThai2.0-WordSeg-Benchmark) ซึ่งเราใช้ตัวคิดคะแนนและชุดทดสอบ 4 ชุด
- โปรแกรม: [iapp-ai บน PyPI](https://pypi.org/project/iapp-ai)
- เปเปอร์ของ model นี้: [LATTE-WordSeg-Lite (arXiv 2611.21300)](https://arxiv.org/abs/2611.21300)
- งานวิจัยต้นทาง: [LATTE: Lattice ATTentive Encoding for Character-based Word Segmentation](https://www.jstage.jst.go.jp/article/jnlp/30/2/30_456/_article), Journal of Natural Language Processing, 2023
- ชุดทดสอบ: UD Thai-TUD และ UD Thai-PUD (Universal Dependencies), Wisesight-160 (PyThaiNLP), VISTEC-TP-TH-2021 (VISTEC) และ [TALPCo](https://github.com/matbahasa/TALPCo) (Nomoto และคณะ, 2018, 2019)
- ตัวเทียบ: [DeepCut](https://github.com/rkcosmos/deepcut)
