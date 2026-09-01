"""Final three-source analysis of the SEEN-46 campaign (2026-09-02).

    python scripts/final_three_source_analysis.py

Consumes only frozen/verified sources, read-only:

* FINAL HUMAN REFERENCE (46 cases), built per the owner's rule:
  - independent two-reviewer consensus where the reviewers agreed;
  - adjudicated_human_reference where they disagreed (owner adjudication of
    the disagreement);
  - owner_adjudicated_after_source_repair for the two repaired cases
    (e004_q2_r6 / e004_q2_r8), whose historical reviews stay preserved as
    stale and are never counted;
* LOCAL MODEL: the 44 frozen SEEN-46 outputs plus the two corrected-rerun
  outputs (CORRECTED_RERUN_2026-09-02.jsonl). The two superseded outputs are
  preserved, registered invalid, and never compared;
* ORIGINAL INSTRUCTOR: the frozen derived verdicts
  (ground_truth_source=original_instructor_grade), derivable cases only.

Nothing is modified anywhere: the review DB is opened mode=ro; every input is
a frozen artifact. No model / provider / OCR / RAG call. Reported per the
standing two-layer rule: no source is declared universally correct.
"""
from __future__ import annotations

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
RERUN_JSONL = RUNS / "CORRECTED_RERUN_2026-09-02.jsonl"
SEEN46_RUN_DIRS = ("dev__all__qwen3-vl-8b-instruct__72e19378d1",
                   "calibration__all__qwen3-vl-8b-instruct__e2a3cfc925")
REPAIRED = ("e004_q2_r6", "e004_q2_r8")
OUT_JSON = RUNS / "FINAL_THREE_SOURCE_2026-09-02.json"
OUT_MD = RUNS / "FINAL_THREE_SOURCE_2026-09-02.md"

VERDICTS = ("invalid", "partially_valid", "valid")
RANK = {v: i for i, v in enumerate(VERDICTS)}


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


def block(pairs: list[tuple[str, str]], name_a: str, name_b: str) -> dict:
    """pairs = (reference, prediction). Same math as the review site's compare
    (full-precision recalls; macro-F1 over classes present on either side)."""
    n = len(pairs)
    agree = sum(1 for a, b in pairs if a == b)
    conf: dict[str, dict[str, int]] = {}
    for a, b in pairs:
        conf.setdefault(a, {}).setdefault(b, 0)
        conf[a][b] += 1
    recalls = []
    for cls in VERDICTS:
        support = sum(1 for a, _ in pairs if a == cls)
        if support:
            recalls.append(sum(1 for a, b in pairs if a == cls and b == cls) / support)
    f1s = []
    for cls in VERDICTS:
        support = sum(1 for a, _ in pairs if a == cls)
        predicted = sum(1 for _, b in pairs if b == cls)
        if not support and not predicted:
            continue
        tp = sum(1 for a, b in pairs if a == b == cls)
        prec = tp / predicted if predicted else 0.0
        rec = tp / support if support else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return {"cases": n, "exact_agreement": agree,
            "exact_agreement_pct": round(100 * agree / n, 1) if n else None,
            f"harmful_overgrades__{name_b}_higher_than_{name_a}":
                sum(1 for a, b in pairs if RANK[b] > RANK[a]),
            f"harmful_undergrades__{name_b}_lower_than_{name_a}":
                sum(1 for a, b in pairs if RANK[b] < RANK[a]),
            "confusion_matrix__rows_reference_cols_prediction": conf,
            "balanced_accuracy": round(sum(recalls) / len(recalls), 4) if recalls else None,
            "macro_f1": round(sum(f1s) / len(f1s), 4) if f1s else None}


def by_group(pairs_by_case: dict[str, tuple[str, str]], keyfn) -> dict:
    groups: dict[str, list] = {}
    for cid, p in pairs_by_case.items():
        groups.setdefault(keyfn(cid), []).append(p)
    return {k: {"cases": len(v), "agree": sum(1 for a, b in v if a == b),
                "agree_pct": round(100 * sum(1 for a, b in v if a == b) / len(v), 1)}
            for k, v in sorted(groups.items())}


def main() -> int:
    # ------------------------------------------------ frozen inputs ---------
    items = json.loads((BUNDLE / "items.json").read_text(encoding="utf-8"))
    id_map = json.loads((BUNDLE / "private" / "id_map.json").read_text(encoding="utf-8"))
    fp_of = {i["item_id"]: i.get("evidence_sha256") for i in items}
    iid_of = {v: k for k, v in id_map.items()}
    ref_raw = json.loads((BUNDLE / "private" / "instructor_reference.json").read_text(encoding="utf-8"))
    instructor = {id_map[i]: v for i, v in ref_raw.items()}

    model: dict[str, dict] = {}
    for d in SEEN46_RUN_DIRS:
        for s in json.loads((RUNS / "grade_primary" / d / "scored.jsonl.json").read_text(encoding="utf-8")):
            if s["case_id"] not in REPAIRED:            # superseded outputs never compared
                model[s["case_id"]] = {**s, "output_source": "frozen_seen46_run"}
    for line in RERUN_JSONL.read_text(encoding="utf-8").splitlines():
        s = json.loads(line)
        model[s["case_id"]] = {**s, "output_source": "corrected_rerun_2026-09-02"}
    assert len(model) == 46, len(model)
    assert all(model[c]["output_source"] == "corrected_rerun_2026-09-02" for c in REPAIRED)

    # ------------------------------- the final human reference (46) ---------
    labels = _ro("SELECT * FROM labels WHERE status='saved'")
    finals = {f["item_id"]: f for f in _ro("SELECT * FROM final_labels")}
    fresh_by_item: dict[str, list] = {}
    stale_by_item: dict[str, list] = {}
    for l in labels:
        target = (stale_by_item if (l.get("evidence_sha256") and fp_of.get(l["item_id"])
                                    and l["evidence_sha256"] != fp_of[l["item_id"]])
                  else fresh_by_item)
        target.setdefault(l["item_id"], []).append(l)

    human: dict[str, dict] = {}
    redundant_finals = []
    for i in items:
        iid, cid = i["item_id"], id_map[i["item_id"]]
        fresh = [( l["grader"], _note(l["note"]).get("verdict")) for l in fresh_by_item.get(iid, [])]
        fin = finals.get(iid)
        fkind = _note(fin["note"]).get("kind") if fin else None
        fverdict = _note(fin["note"]).get("verdict") if fin else None
        if fkind == "owner_adjudicated_after_source_repair":
            assert cid in REPAIRED and not fresh and len(stale_by_item.get(iid, [])) == 2
            human[cid] = {"verdict": fverdict, "source": "owner_adjudicated_after_source_repair"}
        elif len(fresh) == 2 and len({v for _, v in fresh}) == 1:
            human[cid] = {"verdict": fresh[0][1], "source": "independent_two_reviewer_consensus"}
            if fin is not None:                        # redundant confirming final
                assert fverdict == fresh[0][1], (cid, fverdict, fresh)
                human[cid]["redundant_confirming_adjudication"] = True
                redundant_finals.append(cid)
        elif fin is not None:
            human[cid] = {"verdict": fverdict, "source": "adjudicated_human_reference"}
        else:
            raise AssertionError(f"unresolved case {cid}")
    assert len(human) == 46
    src_counts: dict[str, int] = {}
    for h in human.values():
        src_counts[h["source"]] = src_counts.get(h["source"], 0) + 1
    assert src_counts == {"independent_two_reviewer_consensus": 22,
                          "adjudicated_human_reference": 22,
                          "owner_adjudicated_after_source_repair": 2}, src_counts

    # ------------------- reviewer agreement BEFORE adjudication (44) --------
    pre_pairs = []
    for iid, ls in fresh_by_item.items():
        if len(ls) == 2:
            a, b = sorted(ls, key=lambda l: l["grader"])
            pre_pairs.append((_note(a["note"]).get("verdict"), _note(b["note"]).get("verdict")))
    n44 = len(pre_pairs)
    agree44 = sum(1 for a, b in pre_pairs if a == b)
    pa = {v: sum(1 for a, _ in pre_pairs if a == v) / n44 for v in VERDICTS}
    pb = {v: sum(1 for _, b in pre_pairs if b == v) / n44 for v in VERDICTS}
    po = agree44 / n44
    pe = sum(pa[v] * pb[v] for v in VERDICTS)
    kappa = (po - pe) / (1 - pe) if pe != 1 else None

    # --------------------------------------------- the four blocks ----------
    hv = {c: h["verdict"] for c, h in human.items()}
    mv = {c: m["predicted_verdict"] for c, m in model.items()}
    iv = {c: r["instructor_derived_verdict"] for c, r in instructor.items()
          if r.get("instructor_derived_verdict")}

    A_pairs = {c: (hv[c], mv[c]) for c in hv}
    B_pairs = {c: (hv[c], iv[c]) for c in hv if c in iv}
    C_pairs = {c: (iv[c], mv[c]) for c in iv}
    D = {"all_agree": 0, "human_model_agree_only": 0, "human_instructor_agree_only": 0,
         "model_instructor_agree_only": 0, "all_disagree": 0, "cases": 0}
    for c in hv:
        if c in iv:
            h, m_, s = hv[c], mv[c], iv[c]
            D["cases"] += 1
            if h == m_ == s:
                D["all_agree"] += 1
            elif h == m_:
                D["human_model_agree_only"] += 1
            elif h == s:
                D["human_instructor_agree_only"] += 1
            elif m_ == s:
                D["model_instructor_agree_only"] += 1
            else:
                D["all_disagree"] += 1

    lenient = sum(1 for h, i in B_pairs.values() if RANK[i] > RANK[h])
    strict = sum(1 for h, i in B_pairs.values() if RANK[i] < RANK[h])

    ev = {
        "outputs": len(model),
        "auto": sum(1 for m in model.values() if m["decision"] == "AUTO"),
        "auto_rate_pct": round(100 * sum(1 for m in model.values() if m["decision"] == "AUTO") / len(model), 1),
        "validation_ok": sum(1 for m in model.values() if m.get("validation_ok")),
        "evidence_failures": sum(1 for m in model.values() if m.get("evidence_failure")),
        "ungrounded_invalid": sum(1 for m in model.values() if m.get("evidence_ungrounded_invalid")),
        "schema_failures": sum(1 for m in model.values() if m.get("schema_failure")),
        "uncertain": sum(1 for m in model.values() if m.get("uncertain")),
    }

    writer = lambda c: c.split("_")[0]                                   # noqa: E731
    question = lambda c: c.split("_")[1]                                 # noqa: E731

    doc = {
        "artifact": "final_three_source_analysis",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "policy": ["instructor grade = reference, not infallible truth; humans are not "
                   "infallible either; no source is declared universally correct",
                   "seen exams only (DEV+CALIBRATION) - no generalization claim; HELD_OUT sealed",
                   "the repaired cases' human reference is the owner's post-repair verdict, "
                   "NEVER two-reviewer consensus; their superseded model outputs are preserved "
                   "and excluded"],
        "final_human_reference": {
            "cases": 46, "by_source": src_counts,
            "redundant_confirming_adjudications": redundant_finals,
            "per_case": {c: human[c] for c in sorted(human)},
        },
        "model_output_sources": {"frozen_seen46_run": 44, "corrected_rerun_2026-09-02": 2},
        "reviewer_agreement_before_adjudication": {
            "cases_with_two_fresh_reviews": n44, "agreements": agree44,
            "agreement_pct": round(100 * po, 1), "cohens_kappa": round(kappa, 4),
            "note": "the 44 unaffected cases only; the repaired cases' stale reviews are "
                    "preserved but never counted",
        },
        "A_model_vs_final_human_reference": {
            **block(list(A_pairs.values()), "human", "model"),
            "per_writer": by_group(A_pairs, writer),
            "per_question": by_group(A_pairs, question),
        },
        "B_instructor_vs_final_human_reference": {
            **block(list(B_pairs.values()), "human", "instructor"),
            "derivable_instructor_verdicts": len(iv),
            "cases_without_derivable_instructor_verdict": len(hv) - len(B_pairs),
            "instructor_more_lenient_than_human": lenient,
            "instructor_stricter_than_human": strict,
            "per_writer": by_group(B_pairs, writer),
            "per_question": by_group(B_pairs, question),
        },
        "C_model_vs_original_instructor": {
            **block(list(C_pairs.values()), "instructor", "model"),
            "per_writer": by_group(C_pairs, writer),
            "per_question": by_group(C_pairs, question),
        },
        "D_three_way_agreement": D,
        "model_evidence_validation": ev,
        "repaired_cases_detail": {
            c: {"human_reference": human[c],
                "model_corrected": {k: model[c][k] for k in
                                    ("predicted_verdict", "score", "decision",
                                     "evidence_failure", "cache_hit", "latency_s")},
                "instructor_derived_verdict": iv.get(c),
                "instructor_score": instructor[c]["actual_instructor_score"]}
            for c in REPAIRED},
        "confirmations": {
            "new_local_model_calls": 2, "cloud_calls": 0, "ocr_calls": 0, "rag_calls": 0,
            "held_out_calls_or_exposure": 0,
            "instructor_grades_modified": 0, "human_reviews_modified": 0,
            "historical_model_outputs_modified": 0, "prompt_modified": 0,
        },
    }
    OUT_JSON.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8",
                        newline="\n")

    A, B, C = (doc["A_model_vs_final_human_reference"],
               doc["B_instructor_vs_final_human_reference"],
               doc["C_model_vs_original_instructor"])
    md = [
        f"# Final three-source analysis — SEEN-46 ({doc['created_at']})", "",
        "Final human reference (46) = 22 independent two-reviewer consensus + 22 "
        "adjudicated_human_reference (disagreements) + 2 owner_adjudicated_after_source_repair "
        "(e004_q2_r6/r8). Model = 44 frozen SEEN-46 outputs + 2 corrected-rerun outputs "
        "(superseded pair preserved, registered invalid, excluded). Instructor = frozen "
        "derived verdicts (derivable cases only). No source is declared universally correct.", "",
        f"- reviewer agreement before adjudication: **{agree44}/{n44} = {round(100*po,1)}%**, "
        f"Cohen's kappa **{round(kappa, 3)}**",
        f"- A model vs final human reference: **{A['exact_agreement']}/{A['cases']} = "
        f"{A['exact_agreement_pct']}%** (macro-F1 {A['macro_f1']}, balanced acc "
        f"{A['balanced_accuracy']}; overgrades {A['harmful_overgrades__model_higher_than_human']}, "
        f"undergrades {A['harmful_undergrades__model_lower_than_human']})",
        f"- B instructor vs final human reference ({B['cases']} derivable): "
        f"**{B['exact_agreement']}/{B['cases']} = {B['exact_agreement_pct']}%** "
        f"(instructor more lenient {lenient}, stricter {strict})",
        f"- C model vs original instructor ({C['cases']} derivable): "
        f"**{C['exact_agreement']}/{C['cases']} = {C['exact_agreement_pct']}%** "
        f"(macro-F1 {C['macro_f1']}; overgrades "
        f"{C['harmful_overgrades__model_higher_than_instructor']}, undergrades "
        f"{C['harmful_undergrades__model_lower_than_instructor']})",
        f"- D three-way over {D['cases']} cases: all {D['all_agree']}, human+model "
        f"{D['human_model_agree_only']}, human+instructor {D['human_instructor_agree_only']}, "
        f"model+instructor {D['model_instructor_agree_only']}, none {D['all_disagree']}",
        f"- model evidence validation: AUTO {ev['auto']}/{ev['outputs']} "
        f"({ev['auto_rate_pct']}%), evidence failures {ev['evidence_failures']}, "
        f"schema failures {ev['schema_failures']}", "",
        "Repaired cases (owner reference, corrected model output):", "",
        "| case | owner verdict | corrected model | instructor (derived) | instructor score |",
        "|---|---|---|---|---|",
    ]
    for c in REPAIRED:
        r = doc["repaired_cases_detail"][c]
        md.append(f"| {c} | {r['human_reference']['verdict']} | "
                  f"{r['model_corrected']['predicted_verdict']} | "
                  f"{r['instructor_derived_verdict'] or '-'} | {r['instructor_score']:g} |")
    OUT_MD.write_text("\n".join(md) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: doc[k] for k in ("final_human_reference", "reviewer_agreement_before_adjudication",
                                          "D_three_way_agreement", "model_evidence_validation",
                                          "confirmations")}, ensure_ascii=False, indent=1,
                     default=str)[:2000])
    print("written:", OUT_JSON.name, OUT_MD.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
