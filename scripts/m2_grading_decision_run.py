"""Downstream grading-DECISION-PRESERVATION experiment (Mission 2, P2/P3).

Measures: how often OCR corruption changes the FIXED grader's decision.
It does NOT measure objectively correct grading — decision_ref is only
what the same fixed judge decides on the owner-verified reference text.

Frozen design (from scripts/m2_grading_eval.py, unchanged): local
qwen3-vl judge, temperature 0, categorical verdicts (valid /
partially_valid / invalid / unintelligible), printed question context
from the born-digital booklet, judge calls cached by (cell, sha1(text)) —
identical context/key/grader between arms; ONLY the explanation text
differs.

Eligibility per backend: every benchmark cell whose OWNER reference is
strict (no unreadable spans), >= 25 chars, and whose constituent items
ALL have valid persisted predictions in that backend's arm. Exclusions
are recorded with reasons. Per-cell results append to
evaluation/m2_grading/<config>.jsonl immediately (resumable); the gold
verdict ledger is shared across backends via the cache.

Serialize runs — one Ollama job at a time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "evaluation" / "hebrew_bench_v2"
OUT = REPO / "evaluation" / "m2_grading"

import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location("mge", REPO / "scripts" / "m2_grading_eval.py")
mge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mge)
spec2 = importlib.util.spec_from_file_location("hb", REPO / "scripts" / "hebrew_bench_eval.py")
hb = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(hb)

VERDICT_ORDER = {"valid": 2, "partially_valid": 1, "invalid": 0, "unintelligible": None}


def cell_items() -> dict[str, list[str]]:
    items = json.loads((BENCH / "items.json").read_text(encoding="utf-8"))["items"]
    cells = defaultdict(list)
    for it in items:
        iid = it["id"]
        if iid.startswith("hl_"):
            cells[iid.split("__")[0].replace("hl_", "")].append(iid)
        elif iid.startswith("hc_"):
            cells[iid.replace("hc_", "")].append(iid)
    return dict(cells)


def strict_cell_refs() -> dict[str, str]:
    refs = json.loads((BENCH / "references.json").read_text(encoding="utf-8"))
    items = {i["id"]: i for i in json.loads((BENCH / "items.json").read_text(encoding="utf-8"))["items"]}
    out = {}
    for cell, iids in cell_items().items():
        if any(items[i].get("hard") for i in iids):
            continue  # reference itself incomplete -> excluded (recorded later)
        text = " ".join(refs[i]["text"] for i in sorted(iids)).strip()
        out[cell] = text
    return out


def backend_cells(config: str) -> tuple[dict[str, str], dict[str, str]]:
    """cell -> joined OCR text for cells FULLY covered by the arm; plus
    exclusion reasons for cells that are not eligible."""
    outdir = BENCH / "outputs" / config / "run1"
    preds = {}
    for f in outdir.glob("*.json"):
        r = json.loads(f.read_text(encoding="utf-8"))
        if not r.get("error"):
            preds[r["item"]] = r.get("transcription") or ""
    refs = strict_cell_refs()
    eligible, excluded = {}, {}
    items_meta = {i["id"]: i for i in json.loads((BENCH / "items.json").read_text(encoding="utf-8"))["items"]}
    for cell, iids in cell_items().items():
        if cell not in refs:
            excluded[cell] = "hard reference (contains unreadable spans)"
            continue
        if len(refs[cell]) < 25:
            excluded[cell] = "reference under 25 chars (not a substantive explanation)"
            continue
        missing = [i for i in iids if i not in preds]
        if missing:
            excluded[cell] = f"backend lacks predictions for {len(missing)}/{len(iids)} items"
            continue
        eligible[cell] = " ".join(preds[i] for i in sorted(iids)).strip()
    return eligible, excluded


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="benchmark arm, e.g. qwen8b_strict_contrast")
    ap.add_argument("--base-url", default="http://localhost:11434/v1")
    ap.add_argument("--judge-model", default="qwen3-vl:8b-instruct")
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    ledger_path = OUT / f"{args.config}.jsonl"
    done = set()
    if ledger_path.exists():
        for line in ledger_path.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(line)["cell"])
            except json.JSONDecodeError:
                pass

    refs = strict_cell_refs()
    hyps, excluded = backend_cells(args.config)
    (OUT / f"{args.config}_exclusions.json").write_text(
        json.dumps(excluded, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    ctx = mge.question_context()
    todo = [c for c in sorted(hyps) if c not in done]
    print(f"[{args.config}] eligible cells: {len(hyps)} | excluded: {len(excluded)} "
          f"| already done: {len(done)} | todo: {len(todo)}")

    for n, cell in enumerate(todo, 1):
        t0 = time.monotonic()
        try:
            gv = mge.judge(cell, refs[cell], ctx, args.base_url, args.judge_model)
            ov = mge.judge(cell, hyps[cell], ctx, args.base_url, args.judge_model)
        except Exception as e:  # noqa: BLE001 — log and continue; resume later
            print(f"  {cell}: JUDGE ERROR {type(e).__name__}: {str(e)[:120]}")
            continue
        ref_norm, hyp_norm = hb.normalize(refs[cell]), hb.normalize(hyps[cell])
        cer = hb.lev(hyp_norm, ref_norm) / max(len(ref_norm), 1)
        rec = {
            "cell": cell,
            "writer": cell.split("_")[0],
            "ref_sha1": hashlib.sha1(refs[cell].encode()).hexdigest(),
            "ref_text": refs[cell],
            "ocr_text": hyps[cell],
            "ocr_cell_cer": round(cer, 3),
            "verdict_ref": gv["verdict"],
            "verdict_ocr": ov["verdict"],
            "reason_ref": gv.get("reason", ""),
            "reason_ocr": ov.get("reason", ""),
            "agree": gv["verdict"] == ov["verdict"],
            "direction": (
                "same" if gv["verdict"] == ov["verdict"]
                else "abstain" if VERDICT_ORDER[ov["verdict"]] is None
                else "indeterminate" if VERDICT_ORDER[gv["verdict"]] is None
                else "up" if VERDICT_ORDER[ov["verdict"]] > VERDICT_ORDER[gv["verdict"]]
                else "down"
            ),
            "latency_s": round(time.monotonic() - t0, 1),
        }
        with ledger_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"  [{n}/{len(todo)}] {cell}: ref={rec['verdict_ref']} ocr={rec['verdict_ocr']} "
              f"{'AGREE' if rec['agree'] else rec['direction'].upper()} "
              f"cer={rec['ocr_cell_cer']} ({rec['latency_s']}s)")
    print(f"[{args.config}] complete: ledger {ledger_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
