"""Iteration-7 pipeline: Surya OCR (multilingual document OCR) on the bench.

Runs INSIDE .venv-htr, fully local. Surya performs its own line detection +
recognition (page-level input), so the cell crops are passed whole — the
complete pipeline is what gets evaluated. Raw output only. Ground truth is
never read here. Handles the known Surya API variants defensively and
records the resolved API + model revisions in config.json.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BENCH = Path("evaluation/hebrew_bench")


def load_ocr():
    """Return (name, fn(image)->text) for the installed Surya version."""
    from PIL import Image  # noqa: F401

    try:  # modern (v2-era) API
        from surya.detection import DetectionPredictor
        from surya.recognition import RecognitionPredictor

        det = DetectionPredictor()
        rec = RecognitionPredictor()

        def run(img):
            preds = rec([img], det_predictor=det)
            lines = getattr(preds[0], "text_lines", None) or []
            return " ".join((l.text or "").strip() for l in lines if (l.text or "").strip())

        return "surya.recognition.RecognitionPredictor+DetectionPredictor", run
    except Exception as e1:  # noqa: BLE001
        try:  # foundation-model API variant
            from surya.foundation import FoundationPredictor
            from surya.detection import DetectionPredictor
            from surya.recognition import RecognitionPredictor

            det = DetectionPredictor()
            rec = RecognitionPredictor(FoundationPredictor())

            def run(img):
                preds = rec([img], det_predictor=det)
                lines = getattr(preds[0], "text_lines", None) or []
                return " ".join((l.text or "").strip() for l in lines if (l.text or "").strip())

            return "surya.foundation.FoundationPredictor variant", run
        except Exception as e2:  # noqa: BLE001
            raise RuntimeError(f"no known Surya API worked: {e1!r} / {e2!r}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-id", default="it7_surya")
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--cells", default="")
    args = ap.parse_args()

    from PIL import Image

    api_name, run_ocr = load_ocr()
    print(f"surya API: {api_name}")

    manifest = json.loads((BENCH / "crops_manifest.json").read_text(encoding="utf-8"))
    if args.cells:
        keep = set(args.cells.split(","))
        manifest = [m for m in manifest if m["id"] in keep]
    outdir = BENCH / "outputs" / args.config_id
    outdir.mkdir(parents=True, exist_ok=True)
    import surya

    meta = {
        "config_id": args.config_id,
        "model": f"surya-ocr {getattr(surya, '__version__', '?')} ({api_name})",
        "prompt": "n/a (document OCR)", "preproc": "none (whole cell; surya does detection)",
        "max_tokens": 0, "runs": args.runs, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (outdir / "config.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")

    t0 = time.monotonic()
    for run in range(1, args.runs + 1):
        rundir = outdir / f"run{run}"
        rundir.mkdir(exist_ok=True)
        for cell in manifest:
            target = rundir / f"{cell['id']}.json"
            if target.exists():
                continue
            img = Image.open(io.BytesIO(Path(cell["file"]).read_bytes())).convert("RGB")
            t1 = time.monotonic()
            try:
                text = run_ocr(img)
                err = None
            except Exception as e:  # noqa: BLE001
                text, err = "", f"{type(e).__name__}: {e}"
            dt = time.monotonic() - t1
            target.write_text(json.dumps({
                "cell": cell["id"], "run": run, "raw": text, "transcription": text,
                "error": err, "latency_s": round(dt, 2),
            }, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"[{args.config_id} run{run}] {cell['id']}: {dt:.1f}s len={len(text)}")
    meta["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    meta["total_wall_s"] = round(time.monotonic() - t0, 1)
    (outdir / "config.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"done in {meta['total_wall_s']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
