"""Phase 2B orchestrator: ML Kit Digital Ink smoke over the Android emulator.

Drives the android/mlkit-ink-runner app via adb: pushes the Phase-2A.1
rtl_a1 synthetic stroke JSONs for the 5 PRE-RECORDED smoke items, launches
the app (which downloads the Hebrew model once, then recognizes fully
on-device), polls for completion, and pulls results into the standard
benchmark output layout:

  evaluation/hebrew_bench_v2/outputs/mlkit_ink_rtl_a1/run1/<item>.json

Inference-side code — this script and the app — never reads references.json.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "evaluation" / "hebrew_bench_v2"
STROKES = BENCH / "ink_strokes" / "rtl_a1"
OUTDIR = BENCH / "outputs" / "mlkit_ink_rtl_a1" / "run1"
ADB = r"C:\Users\ethan\android-m2\sdk\platform-tools\adb.exe"
PKG = "com.m2.inkrunner"
TMP_IN = "/data/local/tmp/m2in"

# Chosen a priori and recorded in the Phase-2B plan BEFORE any reference
# access: clean e004, clean e005, improved thick e003, e006, difficult
# strict e002 cell.
SMOKE_ITEMS = [
    "hl_e004_q1_r3__l1",
    "hl_e005_q1_r2__l1",
    "hl_e003_q1_r1__l1",
    "hl_e006_q1_r3__l1",
    "hc_e002_q1_r2",
]


def adb(*args, check=True, timeout=120):
    r = subprocess.run([ADB, *args], capture_output=True, text=True, timeout=timeout)
    if check and r.returncode != 0:
        raise RuntimeError(f"adb {' '.join(args)} failed: {r.stderr.strip()[:300]}")
    return r.stdout


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    cfg = {
        "config_id": "mlkit_ink_rtl_a1",
        "backend": "mlkit_digital_ink",
        "model": "he (Digital Ink Recognition, on-device)",
        "strokes": "rtl_a1 synthetic reconstruction (order approximate)",
        "items": SMOKE_ITEMS,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (OUTDIR.parent / "config.json").write_text(json.dumps(cfg, indent=1),
                                               encoding="utf-8")

    print(adb("devices").strip())
    adb("shell", "mkdir", "-p", TMP_IN)
    for item in SMOKE_ITEMS:
        src = STROKES / f"{item}.json"
        if not src.exists():
            print(f"missing strokes for {item} — run m2_ink_strokes.py --version a1")
            return 2
        adb("push", str(src), f"{TMP_IN}/{item}.json")
    # copy into the app-internal dir (debuggable app -> run-as allowed)
    adb("shell", f"run-as {PKG} sh -c 'mkdir -p files/in files/out && cp {TMP_IN}/*.json files/in/'")
    adb("shell", "am", "force-stop", PKG)
    adb("shell", "am", "start", "-n", f"{PKG}/.MainActivity")

    t0 = time.monotonic()
    while time.monotonic() - t0 < 600:
        time.sleep(5)
        status = adb("shell", f"run-as {PKG} cat files/status.txt", check=False).strip()
        print(f"  status: {status or '(none)'}")
        if status.startswith(("done", "fatal")):
            break
    for item in SMOKE_ITEMS:
        r = subprocess.run(
            [ADB, "exec-out", "run-as", PKG, "cat", f"files/out/{item}.json"],
            capture_output=True, text=True,
        )
        ok = r.returncode == 0 and r.stdout.strip().startswith("{")
        if ok:
            (OUTDIR / f"{item}.json").write_text(r.stdout, encoding="utf-8")
        print(f"pull {item}: {'ok' if ok else (r.stderr.strip() or r.stdout.strip())[:120]}")
        if ok:
            # normalize to the benchmark record schema
            rec = json.loads((OUTDIR / f"{item}.json").read_text(encoding="utf-8"))
            stroke_meta = json.loads((STROKES / f"{item}.json").read_text(encoding="utf-8"))
            rec.update({
                "run": 1,
                "category": "handwritten_cell" if item.startswith("hc_") else "handwritten_line",
                "raw": json.dumps(rec.get("candidates", []), ensure_ascii=False),
                "latency_s": round((rec.get("recognition_ms") or 0) / 1000.0, 2),
                "conversion_latency_s": stroke_meta.get("conversion_latency_s"),
                "n_strokes": stroke_meta.get("n_strokes"),
                "error": rec.get("error"),
            })
            (OUTDIR / f"{item}.json").write_text(
                json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8"
            )
    print("smoke collection complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
