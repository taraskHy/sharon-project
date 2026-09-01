"""Freeze the FINAL 46-case human reference of the SEEN campaign (2026-09-02).

    python scripts/final_reference_freeze.py

Re-derives every reference verdict from the review database (opened mode=ro)
and the served bundle, records the source of every case DISTINCTLY
(two_reviewer_consensus / adjudicated_human_reference /
owner_adjudicated_after_source_repair), preserves full provenance (both blind
reviews, stale historical reviews, adjudication records, the original
instructor grade, the baseline model output pointer, the corrected-r6/r8
chain), cross-checks against the committed FINAL_THREE_SOURCE analysis, and
writes an immutable self-hashed artifact. Also recomputes the class
distribution and EXACT per-class baseline metrics from per-case rows (never
from rounded aggregates), and analyzes the redundant final on e004_q2_r2.

Nothing is modified anywhere. No model / provider / OCR / RAG call.
SEEN cases only (DEV + CALIBRATION); HELD_OUT is structurally absent.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

RUNS = REPO / "evaluation" / "model_selection" / "runs" / "local_grade_primary"
DATA = Path(os.environ["LOCALAPPDATA"], "autograder", "review46")
BUNDLE = DATA / "bundle"
DB = DATA / "labels.db"
THREE_SOURCE = RUNS / "FINAL_THREE_SOURCE_2026-09-02.json"
RERUN_JSONL = RUNS / "CORRECTED_RERUN_2026-09-02.jsonl"
SEEN46_RUN_DIRS = ("dev__all__qwen3-vl-8b-instruct__72e19378d1",
                   "calibration__all__qwen3-vl-8b-instruct__e2a3cfc925")
REPAIRED = ("e004_q2_r6", "e004_q2_r8")
REDUNDANT = "e004_q2_r2"

OUT_REF = RUNS / "FINAL_HUMAN_REFERENCE_2026-09-02.json"
OUT_METRICS = RUNS / "BASELINE_CLASS_METRICS_2026-09-02.json"
OUT_MD = RUNS / "FINAL_HUMAN_REFERENCE_2026-09-02.md"

VERDICTS = ("invalid", "partially_valid", "valid")
RANK = {v: i for i, v in enumerate(VERDICTS)}

# The canonical source labels of this freeze — never collapsed:
SOURCES = ("two_reviewer_consensus", "adjudicated_human_reference",
           "owner_adjudicated_after_source_repair")
# mapping from the analysis artifact's label for the consensus source
_ANALYSIS_SOURCE = {"independent_two_reviewer_consensus": "two_reviewer_consensus",
                    "adjudicated_human_reference": "adjudicated_human_reference",
                    "owner_adjudicated_after_source_repair": "owner_adjudicated_after_source_repair"}


def _ro(sql, args=()):
    c = sqlite3.connect(f"file:{DB.as_posix()}?mode=ro", uri=True)
    c.row_factory = sqlite3.Row
    out = [dict(r) for r in c.execute(sql, args)]
    c.close()
    return out


def _note(raw):
    try:
        return json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {}


def main() -> int:
    items = json.loads((BUNDLE / "items.json").read_text(encoding="utf-8"))
    id_map = json.loads((BUNDLE / "private" / "id_map.json").read_text(encoding="utf-8"))
    fp_of = {i["item_id"]: i.get("evidence_sha256") for i in items}
    ref_raw = json.loads((BUNDLE / "private" / "instructor_reference.json").read_text(encoding="utf-8"))
    instructor = {id_map[i]: v for i, v in ref_raw.items()}

    model: dict[str, dict] = {}
    for d in SEEN46_RUN_DIRS:
        for s in json.loads((RUNS / "grade_primary" / d / "scored.jsonl.json").read_text(encoding="utf-8")):
            if s["case_id"] not in REPAIRED:
                model[s["case_id"]] = {**s, "output_source": f"frozen_seen46_run:{d}"}
    for line in RERUN_JSONL.read_text(encoding="utf-8").splitlines():
        s = json.loads(line)
        model[s["case_id"]] = {**s, "output_source": "corrected_rerun_2026-09-02"}
    assert len(model) == 46

    labels = _ro("SELECT * FROM labels WHERE status='saved'")
    finals = {f["item_id"]: f for f in _ro("SELECT * FROM final_labels")}
    fresh_by_item: dict[str, list] = {}
    stale_by_item: dict[str, list] = {}
    for l in labels:
        is_stale = (l.get("evidence_sha256") and fp_of.get(l["item_id"])
                    and l["evidence_sha256"] != fp_of[l["item_id"]])
        (stale_by_item if is_stale else fresh_by_item).setdefault(l["item_id"], []).append(l)

    def review_record(l):
        n = _note(l["note"])
        return {"reviewer": l["grader"], "verdict": n.get("verdict"),
                "confidence": n.get("confidence"), "issue": n.get("issue"),
                "note": n.get("text") or "", "revision": l["revision"],
                "updated_at": l.get("updated_at")}

    cases = []
    for i in sorted(items, key=lambda i: id_map[i["item_id"]]):
        iid, cid = i["item_id"], id_map[i["item_id"]]
        fresh = sorted((review_record(l) for l in fresh_by_item.get(iid, [])),
                       key=lambda r: r["reviewer"])
        stale = sorted((review_record(l) for l in stale_by_item.get(iid, [])),
                       key=lambda r: r["reviewer"])
        fin = finals.get(iid)
        fnote = _note(fin["note"]) if fin else {}
        adj = None
        if fin:
            adj = {"verdict": fnote.get("verdict"), "score": fin["score"],
                   "adjudicator": fin.get("adjudicator"), "note": fnote.get("text") or "",
                   "kind": fnote.get("kind") or "adjudicated_human_reference",
                   "at": fin.get("finalized_at") or fin.get("created_at")}
        if adj and adj["kind"] == "owner_adjudicated_after_source_repair":
            assert cid in REPAIRED and not fresh and len(stale) == 2, cid
            verdict, source = adj["verdict"], "owner_adjudicated_after_source_repair"
        elif len(fresh) == 2 and len({r["verdict"] for r in fresh}) == 1:
            verdict, source = fresh[0]["verdict"], "two_reviewer_consensus"
        elif adj:
            verdict, source = adj["verdict"], "adjudicated_human_reference"
        else:
            raise AssertionError(f"unresolved {cid}")
        row = {
            "case_id": cid, "writer": cid.split("_")[0], "question": cid.split("_")[1],
            "final_verdict": verdict, "reference_source": source,
            "independent_blind_reviews": fresh,
            "stale_historical_reviews": stale,
            "adjudication_record": adj,
            "original_instructor": {
                "score": instructor[cid]["actual_instructor_score"],
                "derived_verdict": instructor[cid].get("instructor_derived_verdict"),
                "ground_truth_source": instructor[cid]["ground_truth_source"],
            },
            "baseline_model_output": {
                "verdict": model[cid]["predicted_verdict"], "score": model[cid]["score"],
                "decision": model[cid]["decision"], "output_source": model[cid]["output_source"],
            },
        }
        if cid == REDUNDANT and source == "two_reviewer_consensus" and adj:
            row["redundant_confirming_adjudication"] = True
        if cid in REPAIRED:
            row["corrected_provenance"] = {
                "dataset_revision": "confirmed_row_transposition (owner-confirmed 2026-09-01)",
                "superseded_model_output": "registered invalid_due_to_confirmed_source_transposition "
                                           "(STALE_MODEL_OUTPUTS_2026-09-01.json), preserved",
                "corrected_model_output": "CORRECTED_RERUN_2026-09-02.jsonl",
            }
        cases.append(row)

    assert len(cases) == 46
    assert {c["reference_source"] for c in cases} <= set(SOURCES)
    by_source = {s: sum(1 for c in cases if c["reference_source"] == s) for s in SOURCES}
    assert by_source == {"two_reviewer_consensus": 22, "adjudicated_human_reference": 22,
                         "owner_adjudicated_after_source_repair": 2}, by_source

    # cross-check against the committed analysis artifact
    three = json.loads(THREE_SOURCE.read_text(encoding="utf-8"))
    for c in cases:
        a = three["final_human_reference"]["per_case"][c["case_id"]]
        assert a["verdict"] == c["final_verdict"], c["case_id"]
        assert _ANALYSIS_SOURCE[a["source"]] == c["reference_source"], c["case_id"]

    dist = {v: sum(1 for c in cases if c["final_verdict"] == v) for v in VERDICTS}
    invalid_cases = [{"case_id": c["case_id"], "reference_source": c["reference_source"]}
                     for c in cases if c["final_verdict"] == "invalid"]

    # ---- the redundant final on e004_q2_r2: reopen impact analysis ----------
    r2 = next(c for c in cases if c["case_id"] == REDUNDANT)
    r2_impact = {
        "case_id": REDUNDANT,
        "situation": "both blind reviewers agreed 'valid'; a final (adjudicated, verdict "
                     "'valid') was ALSO recorded at 2026-09-02 00:51:50 — redundant",
        "consensus_verdict": r2["independent_blind_reviews"][0]["verdict"],
        "final_verdict_in_db": (r2["adjudication_record"] or {}).get("verdict"),
        "effective_reference_verdict": r2["final_verdict"],
        "reference_source_used": r2["reference_source"],
        "reopen_effect": "provenance-only cleanup: the DB final row would be deleted; the "
                         "reference verdict stays 'valid' from two-reviewer consensus; the "
                         "source classification here is ALREADY two_reviewer_consensus, so "
                         "no verdict, no source count, and no metric in any A/B/C/D block "
                         "changes. Numeric change: none.",
        "action_taken": "none - owner decides; never deleted automatically",
    }

    doc = {
        "artifact": "final_human_reference_freeze",
        "campaign": "seen46_2026-08-28",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "policy": ["FROZEN: never edited, never re-derived silently; a later correction "
                   "requires an explicit owner-confirmed revision record",
                   "sources stay distinct forever - never collapsed into one label",
                   "instructor grade preserved separately; it is a reference, not the target "
                   "of this freeze",
                   "SEEN development data only (DEV+CALIBRATION); the invalid class is now "
                   "MEASURED on seen data; it remains UNMEASURED on untouched HELD_OUT"],
        "cases": cases,
        "class_distribution": dist,
        "invalid_class_cases": invalid_cases,
        "by_source": by_source,
        "redundant_final_analysis": r2_impact,
    }
    payload = json.dumps({k: v for k, v in doc.items() if k != "reference_sha256"},
                         ensure_ascii=False, sort_keys=True)
    doc["reference_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
    if OUT_REF.exists():
        old = json.loads(OUT_REF.read_text(encoding="utf-8"))
        if old.get("reference_sha256") != doc["reference_sha256"]:
            print("REFUSED: a different reference freeze already exists")
            return 3
    OUT_REF.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8",
                       newline="\n")

    # ---- EXACT baseline per-class metrics from per-case rows ----------------
    pairs = [(c["final_verdict"], c["baseline_model_output"]["verdict"]) for c in cases]
    per_class = {}
    for cls in VERDICTS:
        support = sum(1 for a, _ in pairs if a == cls)
        predicted = sum(1 for _, b in pairs if b == cls)
        tp = sum(1 for a, b in pairs if a == b == cls)
        rec = tp / support if support else None
        prec = tp / predicted if predicted else None
        f1 = (2 * prec * rec / (prec + rec)
              if prec is not None and rec is not None and (prec + rec) else 0.0)
        per_class[cls] = {"support": support, "predicted": predicted, "tp": tp,
                          "recall": round(rec, 4) if rec is not None else None,
                          "recall_exact": f"{tp}/{support}",
                          "precision": round(prec, 4) if prec is not None else None,
                          "precision_exact": f"{tp}/{predicted}",
                          "f1": round(f1, 4)}
    recalls = [per_class[c]["tp"] / per_class[c]["support"] for c in VERDICTS
               if per_class[c]["support"]]
    f1s = [per_class[c]["f1"] for c in VERDICTS
           if per_class[c]["support"] or per_class[c]["predicted"]]
    agree = sum(1 for a, b in pairs if a == b)
    over = sum(1 for a, b in pairs if RANK[b] > RANK[a])
    under = sum(1 for a, b in pairs if RANK[b] < RANK[a])
    metrics = {
        "artifact": "baseline_class_metrics",
        "created_at": doc["created_at"],
        "reference_sha256": doc["reference_sha256"],
        "arm": "baseline qwen3-vl:8b-instruct one-pass (44 frozen + 2 corrected outputs)",
        "cases": 46,
        "exact_agreement": agree, "exact_agreement_pct": round(100 * agree / 46, 1),
        "per_class": per_class,
        "macro_f1": round(sum(f1s) / len(f1s), 4),
        "balanced_accuracy": round(sum(recalls) / len(recalls), 4),
        "harmful_overgrades": over, "harmful_overgrade_rate": f"{over}/46",
        "harmful_undergrades": under, "harmful_undergrade_rate": f"{under}/46",
    }
    OUT_METRICS.write_text(json.dumps(metrics, ensure_ascii=False, indent=1),
                           encoding="utf-8", newline="\n")

    md = [f"# Final human reference — FROZEN ({doc['created_at']})", "",
          f"sha256 `{doc['reference_sha256'][:16]}…` — 46 cases; sources 22 consensus / "
          "22 adjudicated / 2 owner-repaired (distinct forever).", "",
          f"Class distribution: valid **{dist['valid']}**, partially_valid "
          f"**{dist['partially_valid']}**, invalid **{dist['invalid']}**.",
          f"The invalid class is now MEASURED on seen data ({dist['invalid']} cases: "
          f"{', '.join(c['case_id'] for c in invalid_cases)}); it remains UNMEASURED on "
          "HELD_OUT.", "",
          "Baseline (one-pass 8B) exact per-class:", "",
          "| class | recall | precision | F1 |", "|---|---|---|---|"]
    for cls in ("valid", "partially_valid", "invalid"):
        p = per_class[cls]
        md.append(f"| {cls} | {p['recall_exact']} = {p['recall']} | "
                  f"{p['precision_exact']} = {p['precision']} | {p['f1']} |")
    md += ["", f"exact {agree}/46, macro-F1 {metrics['macro_f1']}, balanced accuracy "
           f"{metrics['balanced_accuracy']}, overgrades {over}, undergrades {under}.", "",
           f"Redundant final on {REDUNDANT}: reopen = provenance-only cleanup, "
           "no numeric change (verdict 'valid' from consensus either way)."]
    OUT_MD.write_text("\n".join(md) + "\n", encoding="utf-8", newline="\n")

    print(json.dumps({"class_distribution": dist, "by_source": by_source,
                      "invalid_class_cases": invalid_cases,
                      "metrics": {k: metrics[k] for k in
                                  ("exact_agreement", "per_class", "macro_f1",
                                   "balanced_accuracy", "harmful_overgrades",
                                   "harmful_undergrades")},
                      "reference_sha256": doc["reference_sha256"]},
                     ensure_ascii=False, indent=1))
    print("written:", OUT_REF.name, OUT_METRICS.name, OUT_MD.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
