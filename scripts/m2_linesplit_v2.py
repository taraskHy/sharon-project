"""mlkit_linesplit_v2 — exactly ONE behavioral change vs v1:
SINGLE-BAND PASSTHROUGH.

If the (unchanged) v1 splitter detects exactly one legitimate band AND no
conservative edge-bleed trim occurred, the item bypasses band cropping /
re-thresholding entirely: the FROZEN BASELINE stroke file
(ink_strokes/rtl_a1/<item>.json) is reused byte-for-byte, guaranteeing
identity with the baseline path. Otherwise the item takes the v1 split
path unchanged (v1 stroke files reused verbatim).

No strike/scribble handling, no threshold changes, no recognizer changes.
v1 outputs and stroke files are never modified. Declared as a separate
arm: outputs/mlkit_linesplit_v2/run1.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "evaluation" / "hebrew_bench_v2"
V1_STROKES = BENCH / "ink_strokes" / "linesplit_v1"
BASE_STROKES = BENCH / "ink_strokes" / "rtl_a1"
OUT_STROKES = BENCH / "ink_strokes" / "linesplit_v2"
OUTDIR = BENCH / "outputs" / "mlkit_linesplit_v2" / "run1"
ADB = r"C:\Users\ethan\android-m2\sdk\platform-tools\adb.exe"
PKG = "com.m2.inkrunner"

spec = importlib.util.spec_from_file_location("mls", REPO / "scripts" / "m2_linesplit.py")
mls = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mls)

GATE_20 = mls.GATE_20


def adb(*args, check=True, timeout=120):
    r = subprocess.run([ADB, *args], capture_output=True, text=True, timeout=timeout)
    if check and r.returncode != 0:
        raise RuntimeError(f"adb {' '.join(args)} failed: {r.stderr.strip()[:200]}")
    return r.stdout


def main() -> int:
    OUT_STROKES.mkdir(parents=True, exist_ok=True)
    route = {}
    for item in GATE_20:
        v1 = json.loads((V1_STROKES / f"{item}.json").read_text(encoding="utf-8"))
        passthrough = v1["n_bands"] == 1 and not v1["trimmed_bands"]
        src = BASE_STROKES / f"{item}.json" if passthrough else V1_STROKES / f"{item}.json"
        rec = json.loads(src.read_text(encoding="utf-8"))
        rec["v2_route"] = "passthrough(frozen-baseline strokes)" if passthrough else "v1-split"
        (OUT_STROKES / f"{item}.json").write_text(json.dumps(rec, ensure_ascii=False),
                                                  encoding="utf-8")
        route[item] = rec["v2_route"]
        print(f"{item}: {'PASSTHROUGH' if passthrough else 'SPLIT'} "
              f"(v1 bands={v1['n_bands']}, trimmed={len(v1['trimmed_bands'])})")
    (OUT_STROKES / "routing.json").write_text(json.dumps(route, indent=1), encoding="utf-8")
    if "--convert-only" in sys.argv:
        return 0

    OUTDIR.mkdir(parents=True, exist_ok=True)
    adb("shell", f"run-as {PKG} sh -c 'rm -rf files/in files/out files/status.txt; mkdir -p files/in files/out'")
    adb("shell", "mkdir", "-p", "/data/local/tmp/m2v2")
    for item in GATE_20:
        adb("push", str(OUT_STROKES / f"{item}.json"), f"/data/local/tmp/m2v2/{item}.json")
    adb("shell", f"run-as {PKG} sh -c 'cp /data/local/tmp/m2v2/*.json files/in/'")
    adb("shell", "am", "force-stop", PKG)
    adb("shell", "am", "start", "-n", f"{PKG}/.MainActivity")
    t0 = time.monotonic()
    stable = 0
    while time.monotonic() - t0 < 300:
        time.sleep(5)
        n = adb("shell", f"run-as {PKG} ls files/out", check=False).split()
        status = adb("shell", f"run-as {PKG} cat files/status.txt", check=False).strip()
        if status.startswith("done") and len(n) >= len(GATE_20):
            stable += 1
            if stable >= 2:  # two consecutive confirmations -> no pull race
                break
    got = 0
    for item in GATE_20:
        r = subprocess.run([ADB, "exec-out", "run-as", PKG, "cat", f"files/out/{item}.json"],
                           capture_output=True, text=True)
        if r.returncode == 0 and (r.stdout or "").strip().startswith("{"):
            (OUTDIR / f"{item}.json").write_text(r.stdout, encoding="utf-8")
            got += 1
    print(f"pulled {got}/{len(GATE_20)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
