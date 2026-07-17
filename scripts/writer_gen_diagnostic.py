"""Writer-grouped generalization diagnostic over the pilot TRAIN split.

Protocol pre-registered in evaluation/htr_gen_diag/PROTOCOL.md. Each fold
holds out ALL lines of one writer; the trainer sees an EMPTY val list so
checkpoint selection and early stopping use train loss only — nothing
from the held-out writer can influence them. The per-fold symbol table is
the predeclared base alphabet + that fold's TRAINING label chars only.

    prep    build one fold's workspace (images, lists, syms; val empty)
    eval    post-decode scoring of heldout + trainbase decodes (GT read
            here, strictly after decode files exist)
    report  assemble evaluation/writer_generalization_diagnostic.md

Train/decode between prep and eval use scripts/htr_pilot_train.py
unchanged (fixed architecture/settings from the overfit test).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from scripts.hebrew_bench_eval import lev, normalize, word_align  # noqa: E402
from scripts.htr_annotation_lib import (  # noqa: E402
    UNREADABLE_TOKEN, load_all_annotations, load_samples,
)
from scripts.htr_train_prepare import (  # noqa: E402
    BASE_CHARS, augment, load_line_gray, rng_for, to_syms,
)

ROOT = Path("evaluation/htr_pilot")
DIAG = Path("evaluation/htr_gen_diag")
FOLDS = ("e005", "e004", "e003")  # pre-registered order (largest first)
USABLE_CER = 0.25


def eligible_lines() -> list[tuple[dict, str]]:
    """(sample, transcription) for every training-eligible train line."""
    samples = load_samples(ROOT, "train")
    ann = load_all_annotations(ROOT, "train")
    out = []
    for s in samples:
        rec = ann.get(s["sample_id"])
        if rec is None or rec["status"] != "ok" or not rec["human_verified"]:
            continue
        if UNREADABLE_TOKEN in rec["transcription"]:
            continue
        out.append((s, rec["transcription"]))
    return out


def fold_ws(heldout: str) -> Path:
    return DIAG / f"fold_{heldout}"


def cmd_prep(args) -> int:
    import cv2

    heldout = args.heldout
    ws = fold_ws(heldout)
    lines = eligible_lines()
    train = [(s, t) for s, t in lines if s["writer"] != heldout]
    held = [(s, t) for s, t in lines if s["writer"] == heldout]
    if not train or not held:
        print(f"fold {heldout}: empty side (train {len(train)}, held {len(held)})")
        return 3

    # symbol table: predeclared base + THIS FOLD'S training chars only
    train_chars = set()
    for _s, t in train:
        train_chars.update(t)
    charset = sorted(set(BASE_CHARS) | (train_chars - {" "}))
    syms = ["<ctc>", "<space>"] + charset
    known = set(syms)

    for sub in ("lists", "text", "imgs/train", "imgs/trainbase", "imgs/heldout"):
        (ws / sub).mkdir(parents=True, exist_ok=True)
    (ws / "syms.txt").write_text(
        "\n".join(f"{s} {i}" for i, s in enumerate(syms)) + "\n", encoding="utf-8")

    ids, rows, skipped = [], [], 0
    base_ids = []
    for s, text in train:
        symbols = to_syms(text)
        if any(sym not in known for sym in symbols):
            skipped += 1
            continue
        sid = s["sample_id"]
        base = load_line_gray(ROOT / s["images"]["line"])
        cv2.imwrite(str(ws / "imgs" / "train" / f"{sid}.png"), base)
        cv2.imwrite(str(ws / "imgs" / "trainbase" / f"{sid}.png"), base)
        ids.append(sid)
        base_ids.append(sid)
        rows.append(f"{sid} {' '.join(symbols)}")
        for k in range(1, args.aug + 1):
            aug_id = f"{sid}__aug{k}"
            cv2.imwrite(str(ws / "imgs" / "train" / f"{aug_id}.png"),
                        augment(base, rng_for(sid, k)))
            ids.append(aug_id)
            rows.append(f"{aug_id} {' '.join(symbols)}")
    held_ids = []
    for s, _text in held:
        sid = s["sample_id"]
        cv2.imwrite(str(ws / "imgs" / "heldout" / f"{sid}.png"),
                    load_line_gray(ROOT / s["images"]["line"]))
        held_ids.append(sid)

    (ws / "lists" / "train.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
    (ws / "text" / "train.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
    # EMPTY val: trainer selects on train loss; held-out writer cannot leak
    (ws / "lists" / "val.txt").write_text("", encoding="utf-8")
    (ws / "text" / "val.txt").write_text("", encoding="utf-8")
    (ws / "lists" / "trainbase.txt").write_text(
        "\n".join(base_ids) + "\n", encoding="utf-8")
    (ws / "lists" / "heldout.txt").write_text(
        "\n".join(held_ids) + "\n", encoding="utf-8")
    meta = {
        "heldout_writer": heldout, "aug": args.aug,
        "train_lines": len(base_ids), "train_images": len(ids),
        "heldout_lines": len(held_ids), "skipped_unknown_char": skipped,
        "n_symbols": len(syms), "at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (ws / "fold_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(json.dumps(meta))
    return 0


def _score(ws: Path, decode_name: str, refs: dict[str, str]) -> dict:
    per_line = []
    dec = ws / "decodes" / decode_name
    hyp_by_id = {}
    for line in dec.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        sid, conf, text = (line.split(" ", 2) + ["", ""])[:3]
        if "__aug" in sid:
            continue
        hyp_by_id[sid] = (float(conf), text.strip())
    for sid, ref_raw in refs.items():
        conf, hyp_raw = hyp_by_id.get(sid, (0.0, ""))
        hyp, ref = normalize(hyp_raw), normalize(ref_raw)
        cer = lev(hyp, ref) / max(len(ref), 1)
        gw, hw = ref.split(), hyp.split()
        subs, dels, ins = word_align(gw, hw)
        per_line.append({
            "sample_id": sid, "cer": round(cer, 4),
            "wer": round((subs + dels + ins) / max(len(gw), 1), 4),
            "exact": hyp == ref, "conf": conf,
            "ref_words": len(gw), "hyp_words": len(hw),
            "dels": dels, "ins": ins,
            "ref": ref_raw, "hyp": hyp_raw,
        })
    n = len(per_line)
    gtw = sum(r["ref_words"] for r in per_line)
    hyw = sum(r["hyp_words"] for r in per_line)
    return {
        "n_lines": n,
        "mean_cer": round(sum(r["cer"] for r in per_line) / max(n, 1), 4),
        "median_cer": round(sorted(r["cer"] for r in per_line)[n // 2], 4) if n else None,
        "mean_wer": round(sum(r["wer"] for r in per_line) / max(n, 1), 4),
        "exact_rate": round(sum(r["exact"] for r in per_line) / max(n, 1), 4),
        "usable_rate": round(sum(r["cer"] <= USABLE_CER for r in per_line) / max(n, 1), 4),
        "omission_rate": round(sum(r["dels"] for r in per_line) / max(gtw, 1), 4),
        "insertion_rate": round(sum(r["ins"] for r in per_line) / max(hyw, 1), 4),
        "per_line": per_line,
    }


def cmd_eval(args) -> int:
    heldout = args.heldout
    ws = fold_ws(heldout)
    ann = load_all_annotations(ROOT, "train")
    held_ids = [s for s in (ws / "lists" / "heldout.txt")
                .read_text(encoding="utf-8").splitlines() if s]
    base_ids = [s for s in (ws / "lists" / "trainbase.txt")
                .read_text(encoding="utf-8").splitlines() if s]
    held_refs = {sid: ann[sid]["transcription"] for sid in held_ids}
    base_refs = {sid: ann[sid]["transcription"] for sid in base_ids}

    held = _score(ws, "heldout.txt", held_refs)
    train = _score(ws, "trainbase.txt", base_refs)
    # confidence-vs-CER buckets on the held-out side
    buckets = []
    for lo, hi in ((0.0, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 1.01)):
        rows = [r for r in held["per_line"] if lo <= r["conf"] < hi]
        if rows:
            buckets.append({"conf": f"[{lo},{hi})", "n": len(rows),
                            "mean_cer": round(sum(r["cer"] for r in rows) / len(rows), 4)})
    trials = (ws / "trials.jsonl").read_text(encoding="utf-8").splitlines()
    result = {
        "heldout_writer": heldout,
        "fold_meta": json.loads((ws / "fold_meta.json").read_text(encoding="utf-8")),
        "train_side": {k: v for k, v in train.items() if k != "per_line"},
        "heldout_side": {k: v for k, v in held.items() if k != "per_line"},
        "conf_vs_cer": buckets,
        "trials": [json.loads(t) for t in trials if t.strip()],
        "at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (ws / "eval.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    (ws / "per_line.json").write_text(json.dumps(
        {"heldout": held["per_line"], "trainbase": train["per_line"]},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: result[k] for k in
                      ("heldout_writer", "train_side", "heldout_side")},
                     ensure_ascii=False, indent=1))
    return 0


def cmd_report(args) -> int:
    folds = []
    for w in FOLDS:
        p = fold_ws(w) / "eval.json"
        if p.exists():
            folds.append(json.loads(p.read_text(encoding="utf-8")))
    lines = [
        "# Writer-generalization diagnostic — results",
        "",
        f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')} by "
        "`scripts/writer_gen_diagnostic.py report`. Protocol (pre-registered): "
        "`evaluation/htr_gen_diag/PROTOCOL.md`. This diagnostic trains only on "
        "owner-verified train-split lines and never touches val/internal_test/"
        "held-out exams or exam 002.",
        "",
        "## Summary",
        "",
        "| fold (held-out) | train lines | held lines | train CER | held CER "
        "| held median | held WER | exact | usable | omit | insert |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for f in folds:
        m, tr, he = f["fold_meta"], f["train_side"], f["heldout_side"]
        lines.append(
            f"| {f['heldout_writer']} | {m['train_lines']} (×{m['aug']+1}) "
            f"| {m['heldout_lines']} | {tr['mean_cer']} | **{he['mean_cer']}** "
            f"| {he['median_cer']} | {he['mean_wer']} | {he['exact_rate']} "
            f"| {he['usable_rate']} | {he['omission_rate']} "
            f"| {he['insertion_rate']} |")
    lines += ["", "## Confidence vs CER (held-out side)", ""]
    for f in folds:
        lines.append(f"- **{f['heldout_writer']}**: " + "; ".join(
            f"conf {b['conf']} n={b['n']} CER {b['mean_cer']}"
            for b in f["conf_vs_cer"]))
    lines += ["", "## Representative held-out predictions", ""]
    for f in folds:
        per = json.loads((fold_ws(f["heldout_writer"]) / "per_line.json")
                         .read_text(encoding="utf-8"))["heldout"]
        per = sorted(per, key=lambda r: r["cer"])
        picks = per[:3] + [per[len(per) // 2]] + per[-3:]
        lines.append(f"### held-out {f['heldout_writer']}")
        lines.append("")
        for r in picks:
            lines.append(f"- `{r['sample_id']}` CER {r['cer']} conf {r['conf']:.2f}")
            lines.append(f"  - GT: {r['ref']}")
            lines.append(f"  - PR: {r['hyp'] or '(empty)'}")
        lines.append("")
    out = Path("evaluation/writer_generalization_diagnostic.md")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"report -> {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("prep")
    pr.add_argument("--heldout", choices=FOLDS, required=True)
    pr.add_argument("--aug", type=int, default=3)
    ev = sub.add_parser("eval")
    ev.add_argument("--heldout", choices=FOLDS, required=True)
    sub.add_parser("report")
    args = ap.parse_args()
    return {"prep": cmd_prep, "eval": cmd_eval, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
