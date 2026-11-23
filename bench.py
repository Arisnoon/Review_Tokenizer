"""จับเวลา OpenThai2.0-LATTE-WordSeg-Lite บนเครื่องนี้ ทั้งบน CPU และบนการ์ดจอ แล้วเทียบกัน

    .venv\\Scripts\\python.exe -X utf8 bench.py --device cpu          # ตามที่ package ติดตั้งมา: CPU 1 thread
    .venv-gpu12\\Scripts\\python.exe -X utf8 bench.py --device gpu    # model เดียวกันบนการ์ดจอ ผ่าน CUDA 12
    .venv-dml\\Scripts\\python.exe -X utf8 bench.py --device dml      # model เดียวกันบนการ์ดจอ ผ่าน DirectML
    .venv\\Scripts\\python.exe -X utf8 bench.py --report              # เทียบผล CPU กับผลการ์ดจอที่บันทึกไว้
    .venv\\Scripts\\python.exe -X utf8 bench.py --device cpu --chars 100000    # ลองเร็ว ๆ ไม่บันทึกผล

สิ่งที่วัด
    เวลาที่ใช้ตัดข้อความ data/corpus_1m.txt (1 ล้านตัวอักษร) ระดับ short ใน 4 แบบ คือ เรียกทีละบรรทัด
    และส่งทุกบรรทัดในครั้งเดียวโดยให้ package แบ่ง batch 3 ขนาด แต่ละแบบรันซ้ำ --repeats ครั้ง
    แล้วเก็บครั้งที่เร็วที่สุด เวลาโหลด model ไม่นับรวม

การรันบนการ์ดจอ
    package สร้าง ONNX Runtime session ด้วย CPUExecutionProvider อย่างเดียว สคริปต์นี้จึงแทนที่
    onnxruntime.InferenceSession ชั่วคราว ให้ใส่ provider ของการ์ดจอไว้ก่อน CPU โดยไม่แก้ไฟล์ของ package
    environment ที่รันต้องติดตั้ง onnxruntime-gpu (CUDA) หรือ onnxruntime-directml (DirectML) แทน onnxruntime

ข้อผิดพลาดสองอย่างที่โค้ดนี้ป้องกันไว้
    1. เมื่อการ์ดจอรันไม่สำเร็จ ONNX Runtime จะสร้าง session ใหม่บน CPU แล้วรันต่อเงียบ ๆ ถ้าไม่ตรวจ
       จะได้ตัวเลขของ CPU มาโดยเข้าใจว่าเป็นของการ์ดจอ จึงตรวจ provider ซ้ำหลังรันจริง และไม่บันทึกผลถ้าถอยไป CPU
    2. DLL ของ CUDA และ cuDNN มาจาก package nvidia-* ซึ่งไม่อยู่ใน path ที่ Windows ค้นหา และ ONNX Runtime
       รุ่นเก่าไม่รู้จัก DLL ทุกตัวของ cuDNN รุ่นใหม่ จึงเพิ่มโฟลเดอร์เหล่านั้นเข้า path ก่อน (ดู make_cuda_dlls_findable)

ผลลัพธ์
    results/<device>.json          เวลาของทุกแบบ สเปกเครื่อง และ (เฉพาะการ์ดจอ) node ของ model ไปรันที่ไหน
    results/<device>.tokens.txt    ผลตัดคำของการเรียกทีละบรรทัด ใช้เทียบว่า CPU กับการ์ดจอตัดเหมือนกันแค่ไหน
    โดย <device> คือ cpu, gpu หรือ dml

สคริปต์นี้อาศัยของภายใน package สองอย่าง คือ Tokenizer._segmenter และ Segmenter.batch_chars
ถ้า iapp-ai รุ่นใหม่เปลี่ยนชื่อสองอย่างนี้ ต้องแก้ตาม
"""
from __future__ import annotations

import argparse
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Callable

from common import LINE_MODE, RESULTS, ROOT, load_corpus, mode_label, read_json, use_utf8_console, write_json

# จำนวนตัวอักษร (รวมช่องเติม) ต่อ batch ที่ลอง: 4096 คือค่าเริ่มต้นของ package ซึ่งปรับมาสำหรับ CPU 1 thread
# ขนาดที่ใหญ่กว่าให้การ์ดจอมีงานต่อการเรียกหนึ่งครั้งมากขึ้น
BATCH_CHARS = [4096, 32768, 131072]
DEFAULT_BATCH_CHARS = 4096
# ค่าของ --device -> ชื่อ execution provider ของ ONNX Runtime
GPU_PROVIDERS = {"gpu": "CUDAExecutionProvider", "dml": "DmlExecutionProvider"}
WARMUP_LINES = 200          # จำนวนบรรทัดที่รันทิ้งก่อนจับเวลา (การเรียกครั้งแรกต้องจองหน่วยความจำ)
PROFILE_LINES = 200         # จำนวนบรรทัดที่ใช้ดูว่า node ไปรันที่ไหน


# ---------------------------------------------------------------- สเปกเครื่อง
def run_command(command: list[str]) -> str:
    """รันคำสั่งของระบบ แล้วคืนข้อความที่คำสั่งพิมพ์ออกมา หรือข้อความว่างถ้ารันไม่ได้"""
    try:
        return subprocess.run(command, capture_output=True, text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def hardware() -> dict:
    """เก็บสเปกเครื่องที่ใช้วัด เพื่อบันทึกไว้คู่กับผล

    ชื่อ CPU ถามจาก Windows ผ่าน PowerShell ส่วนการ์ดจอถามจาก nvidia-smi ถ้าถามไม่ได้จะใส่ค่าสำรอง
    """
    cpu = run_command(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"])
    gpu = run_command(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"])
    return {
        "cpu": cpu or platform.processor(),
        "cpu_logical_cores": os.cpu_count(),
        "gpu": gpu.splitlines()[0] if gpu else "not found",
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
    }


# ---------------------------------------------------------------- การย้าย session ไปการ์ดจอ
def make_cuda_dlls_findable(ort) -> None:
    """ทำให้ Windows หา DLL ของ CUDA และ cuDNN ที่ติดตั้งมากับ package nvidia-* เจอ

    DLL พวกนี้อยู่ใน site-packages/nvidia/**/ ซึ่งไม่อยู่ใน path ที่ Windows ค้นหา ort.preload_dlls() ช่วยโหลด
    ให้ล่วงหน้าตามรายชื่อตายตัว แต่ ONNX Runtime รุ่นเก่าไม่รู้จัก DLL ทุกตัวของ cuDNN รุ่นใหม่
    เช่น รุ่น 1.24.4 ไม่โหลด cudnn_engines_tensor_ir64_9.dll ของ cuDNN 9.28 ทำให้ Conv ตัวแรกล้มด้วย
    CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED แล้ว session ถอยไปใช้ CPU การเพิ่มโฟลเดอร์เข้า path แก้ได้ทั้งสองกรณี
    """
    nvidia = Path(ort.__file__).parent.parent / "nvidia"
    folders = sorted({str(dll.parent) for dll in nvidia.glob("**/*.dll")})
    if hasattr(os, "add_dll_directory"):                    # มีเฉพาะบน Windows
        for folder in folders:
            os.add_dll_directory(folder)
    os.environ["PATH"] = os.pathsep.join([*folders, os.environ.get("PATH", "")])
    if hasattr(ort, "preload_dlls"):                        # ONNX Runtime รุ่นเก่ามากไม่มีฟังก์ชันนี้
        ort.preload_dlls()


class GpuSessions:
    """ทำให้ ONNX Runtime session ที่ package สร้างหลังจากนี้ ใช้การ์ดจอก่อน CPU

    package ระบุ providers=["CPUExecutionProvider"] ไว้ในโค้ด จึงแทนที่ onnxruntime.InferenceSession
    ด้วยฟังก์ชันที่เปลี่ยนรายชื่อ provider ตอนสร้าง session วิธีนี้ไม่ต้องแก้ไฟล์ของ package ที่ติดตั้งไว้

    profile_prefix: ถ้าตั้งเป็นข้อความ session ที่สร้างถัดไปจะบันทึก profile ลงไฟล์ที่ขึ้นต้นด้วยข้อความนี้
    """

    def __init__(self, provider: str) -> None:
        """provider: ชื่อ execution provider ของการ์ดจอ เช่น "CUDAExecutionProvider" """
        self.provider = provider
        self.profile_prefix: str | None = None

    def install(self) -> None:
        """แทนที่ onnxruntime.InferenceSession ต้องเรียกก่อนที่ package จะสร้าง session แรก"""
        import onnxruntime as ort

        if self.provider == "CUDAExecutionProvider":
            make_cuda_dlls_findable(ort)
        original = ort.InferenceSession

        def session(path, opts=None, providers=None, **kwargs):
            """สร้าง session เหมือนเดิม แต่ใส่ provider ของการ์ดจอไว้หน้า CPU (CPU เป็นตัวสำรองของ node ที่การ์ดจอไม่รับ)"""
            if opts is not None:
                if self.provider == "DmlExecutionProvider":
                    opts.enable_mem_pattern = False         # ข้อกำหนดของ DirectML
                if self.profile_prefix:
                    opts.enable_profiling = True
                    opts.profile_file_prefix = self.profile_prefix
            return original(path, opts, providers=[self.provider, "CPUExecutionProvider"], **kwargs)

        ort.InferenceSession = session


def require_device(segmenter, device: str, when: str) -> None:
    """หยุดโปรแกรมถ้า session ไม่ได้อยู่บนการ์ดจอที่ขอ

    เมื่อการ์ดจอรันไม่สำเร็จ ONNX Runtime จะสร้าง session ใหม่บน CPU แล้วรันต่อ provider ที่เห็นตอนโหลด model
    จึงอาจไม่ตรงกับตอนรันจริง ต้องถามซ้ำหลังรัน ไม่อย่างนั้นจะบันทึกเวลาของ CPU เป็นของการ์ดจอ
    when: ข้อความบอกว่าตรวจตอนไหน ใช้ในข้อความแจ้ง
    """
    if device == "cpu":
        return
    providers = segmenter.session.get_providers()
    if providers[0] != GPU_PROVIDERS[device]:
        raise SystemExit(f"การ์ดจอใช้ไม่ได้ ({when}): ONNX Runtime ถอยไปใช้ {providers} จึงไม่บันทึกผล")


# ---------------------------------------------------------------- การจับเวลา
def best_time(function: Callable[[], object], repeats: int) -> float:
    """รัน function ซ้ำ repeats ครั้ง แล้วคืนเวลาของครั้งที่เร็วที่สุด (วินาที)

    ใช้ครั้งที่เร็วที่สุดเพราะสิ่งรบกวนจากโปรแกรมอื่นทำให้ช้าลงได้อย่างเดียว ไม่มีทางทำให้เร็วขึ้น
    """
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        times.append(time.perf_counter() - start)
    return min(times)


def time_all_modes(tokenizer, lines: list[str], repeats: int) -> tuple[list[dict], list[list[str]]]:
    """จับเวลาทั้ง 4 แบบ คืน (ผลของแต่ละแบบ, ผลตัดคำของการเรียกทีละบรรทัด)

    ขนาด batch เปลี่ยนโดยตั้ง batch_chars ของ Segmenter ภายใน package แล้วตั้งกลับเป็นค่าเริ่มต้นเมื่อจบ
    """
    segmenter = tokenizer._segmenter
    chars = sum(map(len, lines))
    runs: list[dict] = []
    line_output: list[list[str]] = []

    def record(mode: str, seconds: float) -> None:
        """เก็บผลของหนึ่งแบบ และพิมพ์ให้เห็นระหว่างรัน"""
        runs.append({"mode": mode, "seconds": seconds, "chars_per_second": chars / seconds,
                     "seconds_per_million_chars": seconds * 1_000_000 / chars})
        print(f"{mode_label(mode):<26} {seconds:7.2f} วินาที   {chars / seconds:>10,.0f} ตัวอักษร/วินาที", flush=True)

    def one_line_per_call() -> None:
        """เรียก tokenizer หนึ่งครั้งต่อหนึ่งบรรทัด และเก็บผลตัดคำไว้เทียบระหว่างอุปกรณ์"""
        line_output[:] = [tokenizer.word_tokenize(line) for line in lines]

    record(LINE_MODE, best_time(one_line_per_call, repeats))
    for batch_chars in BATCH_CHARS:
        segmenter.batch_chars = batch_chars
        record(f"batch ({batch_chars:,} chars)", best_time(lambda: tokenizer.word_tokenize_batch(lines), repeats))
    segmenter.batch_chars = DEFAULT_BATCH_CHARS
    return runs, line_output


def node_placement(gpu: GpuSessions, lines: list[str]) -> dict:
    """ดูว่าแต่ละ node ของ model ไปรันบน provider ไหน และใช้เวลาเท่าไร

    สร้าง session ใหม่ที่เปิด profiling รันหนึ่งรอบ แล้วอ่านไฟล์ profile ที่ ONNX Runtime เขียน
    คืน {ชื่อ provider: {"nodes": จำนวน node, "ms": เวลารวม, "ops": {ชนิดของ node: จำนวนครั้งที่ถูกเรียก}}}
    """
    from iapp_ai.nlp.tokenizer.latte import Segmenter

    RESULTS.mkdir(exist_ok=True)
    gpu.profile_prefix = str(RESULTS / "gpu_profile")
    try:
        segmenter = Segmenter()             # session นี้ถูกสร้างผ่าน GpuSessions จึงเปิด profiling
    finally:
        gpu.profile_prefix = None           # session ที่สร้างหลังจากนี้ไม่ต้องบันทึก profile
    segmenter.tokenize_batch(lines)
    profile = Path(segmenter.session.end_profiling())
    events = read_json(profile)
    profile.unlink()                        # ไฟล์ profile ใช้ครั้งเดียว ไม่เก็บ

    placement: dict[str, dict] = {}
    for event in events:
        # เหตุการณ์ที่ลงท้ายด้วย _kernel_time คือเวลาคำนวณจริงของ node หนึ่งตัวในการรันหนึ่งครั้ง
        if event.get("cat") != "Node" or not event["name"].endswith("_kernel_time"):
            continue
        provider = event["args"].get("provider") or "unknown"
        entry = placement.setdefault(provider, {"nodes": set(), "ops": {}, "ms": 0.0})
        entry["nodes"].add(event["name"])
        op_name = event["args"].get("op_name", "?")
        entry["ops"][op_name] = entry["ops"].get(op_name, 0) + 1
        entry["ms"] += event["dur"] / 1000   # dur มีหน่วยเป็นไมโครวินาที
    return {provider: {"nodes": len(entry["nodes"]), "ms": round(entry["ms"], 1),
                       "ops": dict(sorted(entry["ops"].items(), key=lambda item: -item[1]))}
            for provider, entry in placement.items()}


def print_placement(placement: dict) -> None:
    """พิมพ์สรุปว่า provider แต่ละตัวรันกี่ node และใช้เวลาเท่าไร"""
    for provider, entry in placement.items():
        print(f"  {provider:<24} {entry['nodes']:>3} node  {entry['ms']:>8.1f} ms")


def bench(device: str, repeats: int, max_chars: int) -> None:
    """จับเวลาบนอุปกรณ์ที่ขอ พิมพ์ผล และบันทึกลง results/ (ยกเว้นเมื่อ max_chars > 0 ซึ่งเป็นการลองเร็ว ๆ)

    device:    "cpu", "gpu" (CUDA) หรือ "dml" (DirectML)
    repeats:   จำนวนครั้งที่รันแต่ละแบบ เก็บครั้งที่เร็วที่สุด
    max_chars: ใช้เฉพาะบรรทัดแรก ๆ จนครบจำนวนตัวอักษรนี้ (0 = ทั้งไฟล์)
    """
    lines = load_corpus(max_chars)
    chars = sum(map(len, lines))

    gpu = None
    if device != "cpu":
        gpu = GpuSessions(GPU_PROVIDERS[device])
        gpu.install()                       # ต้องทำก่อน import package เพื่อให้ session แรกไปการ์ดจอ
    import onnxruntime as ort
    from iapp_ai.nlp import Tokenizer

    start = time.perf_counter()
    tokenizer = Tokenizer(unit="short")
    load_seconds = time.perf_counter() - start
    segmenter = tokenizer._segmenter
    providers = segmenter.session.get_providers()
    require_device(segmenter, device, "ตอนโหลด model")

    info = hardware()
    print(f"CPU       {info['cpu']} ({info['cpu_logical_cores']} logical cores)")
    print(f"การ์ดจอ   {info['gpu']}")
    print(f"Runtime   ONNX Runtime {ort.__version__}, providers {providers}")
    print(f"ข้อความ   {len(lines):,} บรรทัด {chars:,} ตัวอักษร")
    print(f"โหลด model {load_seconds:.2f} วินาที\n")

    tokenizer.word_tokenize_batch(lines[:WARMUP_LINES])     # รันทิ้งก่อน ไม่นับเวลา
    require_device(segmenter, device, "ตอนรันครั้งแรก")
    runs, line_output = time_all_modes(tokenizer, lines, repeats)
    require_device(segmenter, device, "ระหว่างจับเวลา")

    result = {"device": device, "hardware": info, "onnxruntime": ort.__version__, "providers": providers,
              "threads": 1, "lines": len(lines), "chars": chars, "load_seconds": load_seconds,
              "repeats": repeats, "runs": runs}
    if gpu is not None:
        result["node_placement"] = node_placement(gpu, lines[:PROFILE_LINES])
        print("\nnode ของ model ไปรันที่ไหน:")
        print_placement(result["node_placement"])

    if max_chars:
        print("\nลองกับข้อความบางส่วน จึงไม่บันทึกผล")
        return
    write_json(RESULTS / f"{device}.json", result)
    (RESULTS / f"{device}.tokens.txt").write_text(
        "\n".join("|".join(words) for words in line_output) + "\n", encoding="utf-8", newline="\n")
    print(f"\nบันทึกผลที่ {(RESULTS / f'{device}.json').relative_to(ROOT)}")


# ---------------------------------------------------------------- การเทียบผล
def same_lines(first: Path, second: Path) -> tuple[int, int]:
    """นับบรรทัดที่ผลตัดคำเหมือนกันทุกคำในสองไฟล์ .tokens.txt คืน (จำนวนที่เหมือน, จำนวนบรรทัดทั้งหมด)"""
    a = first.read_text(encoding="utf-8").splitlines()
    b = second.read_text(encoding="utf-8").splitlines()
    return sum(x == y for x, y in zip(a, b)), len(a)


def report() -> None:
    """เทียบผล CPU (results/cpu.json) กับผลการ์ดจอทุกแบบที่บันทึกไว้ พิมพ์เป็นตาราง Markdown"""
    cpu_path = RESULTS / "cpu.json"
    if not cpu_path.exists():
        raise SystemExit(f"ไม่พบ {cpu_path}: รัน bench.py --device cpu ก่อน")
    cpu = read_json(cpu_path)
    devices = [device for device in GPU_PROVIDERS if (RESULTS / f"{device}.json").exists()]
    if not devices:
        raise SystemExit("ยังไม่มีผลของการ์ดจอ: รัน bench.py --device gpu หรือ --device dml ก่อน")

    print(f"CPU: {cpu['hardware']['cpu']} (1 thread), ONNX Runtime {cpu['onnxruntime']}")
    print(f"ข้อความ: {cpu['lines']:,} บรรทัด {cpu['chars']:,} ตัวอักษร")
    for device in devices:
        gpu = read_json(RESULTS / f"{device}.json")
        print(f"\nการ์ดจอ: {gpu['hardware']['gpu']} ผ่าน {gpu['providers'][0]}, ONNX Runtime {gpu['onnxruntime']}\n")
        print("| วิธีเรียก | CPU วินาทีต่อ 1 ล้านตัวอักษร | การ์ดจอ วินาทีต่อ 1 ล้านตัวอักษร "
              "| CPU ตัวอักษร/วินาที | การ์ดจอ ตัวอักษร/วินาที | การ์ดจอเทียบกับ CPU |")
        print("|---|---:|---:|---:|---:|---|")
        for c, g in zip(cpu["runs"], gpu["runs"]):
            ratio = g["chars_per_second"] / c["chars_per_second"]
            verdict = f"เร็วกว่า {ratio:.2f} เท่า" if ratio >= 1 else f"ช้ากว่า {1 / ratio:.2f} เท่า"
            print(f"| {mode_label(c['mode'])} | {c['seconds_per_million_chars']:.1f} "
                  f"| {g['seconds_per_million_chars']:.1f} | {c['chars_per_second']:,.0f} "
                  f"| {g['chars_per_second']:,.0f} | {verdict} |")
        same, total = same_lines(RESULTS / "cpu.tokens.txt", RESULTS / f"{device}.tokens.txt")
        print(f"\nผลตัดคำเหมือนกับ CPU: {same:,} จาก {total:,} บรรทัด ({same / total:.1%})")
        if "node_placement" in gpu:
            print("\nnode ของ model ไปรันที่ไหน:")
            print_placement(gpu["node_placement"])


def main() -> None:
    """อ่านตัวเลือกจาก command line แล้วเลือกว่าจะจับเวลา หรือเทียบผลที่บันทึกไว้"""
    use_utf8_console()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", choices=["cpu", *GPU_PROVIDERS],
                        help="cpu, gpu (CUDA) หรือ dml (DirectML ใช้ได้กับการ์ดจอที่รองรับ DirectX 12)")
    parser.add_argument("--report", action="store_true", help="เทียบผล CPU กับผลการ์ดจอที่บันทึกไว้")
    parser.add_argument("--repeats", type=int, default=3, help="จำนวนครั้งที่รันแต่ละแบบ เก็บครั้งที่เร็วที่สุด")
    parser.add_argument("--chars", type=int, default=0,
                        help="ลองกับข้อความแค่จำนวนตัวอักษรนี้ และไม่บันทึกผล (0 = ทั้งไฟล์)")
    args = parser.parse_args()
    if args.report:
        report()
    elif args.device:
        bench(args.device, args.repeats, args.chars)
    else:
        parser.error("ต้องระบุ --device cpu, --device gpu, --device dml หรือ --report")


if __name__ == "__main__":
    main()
