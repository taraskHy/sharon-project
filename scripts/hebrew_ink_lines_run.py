"""Run the fixed transcription config over PRE-SEGMENTED line crops.

Ablation arm for the student-ink isolation experiment: each cell's line
crops (evaluation/student_ink_isolation_artifacts/lines/<cell>/) are
transcribed one call per line with the SAME prompt, model and decoding
settings as scripts/hebrew_bench_run.py, then joined top-to-bottom with a
space into one cell-level transcription so hebrew_bench_eval.py can score
it. Never reads ground truth.

    python scripts/hebrew_ink_lines_run.py --config-id isol3_tsub_lines \
        --manifest evaluation/student_ink_isolation_artifacts/manifests/manifest_lines.json
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from scripts.hebrew_bench_run import PROMPTS, SCHEMA

BENCH = Path("evaluation/hebrew_bench")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-id", required=True)
    ap.add_argument("--model", default="qwen3-vl:8b-instruct")
    ap.add_argument("--base-url", default="http://localhost:11434/v1")
    ap.add_argument("--prompt", choices=sorted(PROMPTS), default="strict_fidelity")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--manifest", required=True,
                    help="manifest with per-cell 'lines': [png paths]")
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    outdir = BENCH / "outputs" / args.config_id
    meta = {
        "config_id": args.config_id, "model": args.model, "prompt": args.prompt,
        "prompt_text": PROMPTS[args.prompt], "preproc": "pre-segmented lines",
        "max_tokens": args.max_tokens, "runs": args.runs,
        "manifest": args.manifest,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "config.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")

    t0 = time.monotonic()
    n_calls = 0
    with httpx.Client(timeout=600.0) as client:
        for run in range(1, args.runs + 1):
            rundir = outdir / f"run{run}"
            rundir.mkdir(exist_ok=True)
            for cell in manifest:
                target = rundir / f"{cell['id']}.json"
                if target.exists():
                    continue
                raws, parts, errs = [], [], []
                dt_total = 0.0
                for lf in cell["lines"]:
                    png = Path(lf).read_bytes()
                    payload = {
                        "model": args.model,
                        "temperature": 0,
                        "max_tokens": args.max_tokens,
                        "messages": [
                            {"role": "system", "content": PROMPTS[args.prompt]},
                            {"role": "user", "content": [
                                {"type": "image_url", "image_url": {
                                    "url": "data:image/png;base64,"
                                    + base64.standard_b64encode(png).decode()}},
                                {"type": "text", "text": "Transcribe now."},
                            ]},
                        ],
                        "response_format": {"type": "json_schema", "json_schema": {
                            "name": "T", "schema": SCHEMA}},
                    }
                    t1 = time.monotonic()
                    resp = client.post(f"{args.base_url}/chat/completions", json=payload)
                    dt_total += time.monotonic() - t1
                    n_calls += 1
                    data = resp.json()
                    try:
                        raw = data["choices"][0]["message"].get("content") or ""
                        raws.append(raw)
                        parts.append(json.loads(raw).get("transcription") or "")
                    except Exception as e:  # noqa: BLE001
                        raws.append("")
                        errs.append(f"{type(e).__name__}: {e} | body={str(data)[:160]}")
                joined = " ".join(p for p in parts if p).strip()
                target.write_text(json.dumps({
                    "cell": cell["id"], "run": run, "raw": json.dumps(raws, ensure_ascii=False),
                    "transcription": joined if (joined or not errs) else None,
                    "error": "; ".join(errs) or None,
                    "latency_s": round(dt_total, 2), "n_lines": len(cell["lines"]),
                }, ensure_ascii=False, indent=1), encoding="utf-8")
                print(f"[{args.config_id} run{run}] {cell['id']}: "
                      f"{dt_total:.1f}s ({len(cell['lines'])} lines)")
    meta["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    meta["total_wall_s"] = round(time.monotonic() - t0, 1)
    meta["calls"] = n_calls
    (outdir / "config.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"done: {n_calls} calls in {meta['total_wall_s']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
