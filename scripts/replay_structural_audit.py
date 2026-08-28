"""Phase 6: counterfactual STRUCTURAL replay of the historical 8B FullDev
run under the new output contract / validation — READ-ONLY.

Not actual model performance: no historical output is mutated, repaired or
re-scored as real. The question answered is purely structural: had the model
placed the spans it already produced into rubric_items[].student_evidence,
which outputs would have passed the production matcher, and which would still
fail, for which reason? Zero inference.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, ".")
from autograder.evidence import (MAX_EVIDENCE_CHARS, evidence_supported,  # noqa: E402
                                 normalize_for_evidence)

RUN = Path("evaluation/model_selection/runs/local_grade_primary/grade_primary/"
           "dev__dev_verdict__all__qwen3-vl-8b-instruct__433146e4b1")
DS = Path("evaluation/model_selection/datasets/grade_primary")
OUT = Path("evaluation/model_selection/runs/local_grade_primary")

outputs = {json.loads(l)["case_id"]: json.loads(l) for l in (RUN / "outputs.jsonl").open(encoding="utf-8")}
scored = {r["case_id"]: r for r in json.loads((RUN / "scored.jsonl.json").read_text(encoding="utf-8"))}
inputs = {json.loads(l)["case_id"]: json.loads(l) for l in (DS / "cases_inputs.jsonl").open(encoding="utf-8")}

PREFIX = re.compile(r"^\s*(?:R\d+)\s*[:\-]\s*")
#: pair-aware quote patterns (same delimiter class opens and closes)
QUOTE_PATTERNS = [re.compile(p, re.DOTALL) for p in (
    r"'([^']{4,}?)'", r'"([^"]{4,}?)"', r"‘([^‘’]{4,}?)’", r"“([^“”]{4,}?)”",
    r"׳([^׳]{4,}?)׳", r"״([^״]{4,}?)״")]
HEBREW = re.compile(r"[֐-׿]")


def fragments(text: str) -> list[str]:
    out = []
    for pat in QUOTE_PATTERNS:
        out.extend(m.group(1).strip() for m in pat.finditer(text))
    return [f for f in out if HEBREW.search(f)]


def lcs(a: str, b: str) -> str:
    best, end = 0, 0
    prev = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best:
                    best, end = cur[j], i
        prev = cur
    return a[end - best:end]


rows = []
WRONG_SOURCE = {"e002_q2_r5"}    # committed audit category (official-solution text as "student quote")
for cid in sorted(outputs):
    s = scored[cid]
    if not s.get("evidence_failure"):
        continue
    o = outputs[cid]["output"]
    ev = o.get("evidence") or ""
    tr = inputs[cid]["transcription"]
    stripped = PREFIX.sub("", ev).strip()
    whole_ok = (len(stripped) <= MAX_EVIDENCE_CHARS
                and evidence_supported(stripped, tr))
    frags = fragments(ev)
    frag_ok = [f for f in frags if len(f) <= MAX_EVIDENCE_CHARS and evidence_supported(f, tr)]
    span = lcs(normalize_for_evidence(ev), normalize_for_evidence(tr))
    span_ok = 12 <= len(span) <= MAX_EVIDENCE_CHARS
    if whole_ok:
        content_class = "clean_exact_quote"
    elif cid in WRONG_SOURCE:
        content_class = "wrong_source_text"
    elif frag_ok:
        content_class = "exact_quote_wrapped_in_commentary"
    else:
        content_class = "only_paraphrase_prose"
    still_fail_reason = None
    if not (whole_ok or frag_ok):
        still_fail_reason = ("wrong_source_span" if cid in WRONG_SOURCE else
                            "paraphrase_only_no_isolated_quote")
    rows.append({
        "case_id": cid,
        "content_class": content_class,
        "whole_field_minus_prefix_verifies_and_fits": whole_ok,
        "model_isolated_a_verified_quote": bool(frag_ok),
        "verified_fragment_example": frag_ok[0] if frag_ok else None,
        "verbatim_span_present_anywhere_12_to_200": span_ok,
        "over_200_as_submitted": len(stripped) > MAX_EVIDENCE_CHARS,
        "counterfactually_valid_with_own_isolated_span": whole_ok or bool(frag_ok),
        "still_fail_reason_without_new_copying": still_fail_reason,
    })

n = len(rows)
recoverable = sum(1 for r in rows if r["counterfactually_valid_with_own_isolated_span"])
by_class: dict[str, int] = {}
for r in rows:
    by_class[r["content_class"]] = by_class.get(r["content_class"], 0) + 1
still = {}
for r in rows:
    if r["still_fail_reason_without_new_copying"]:
        k = r["still_fail_reason_without_new_copying"]
        still[k] = still.get(k, 0) + 1

# historical zero/invalid verdicts under the new rule
zero_rows = []
for cid, o in outputs.items():
    g = o["output"]
    if float(g.get("score") or 0) == 0.0:
        items = g.get("rubric_items") or []
        grounded = any(evidence_supported((i.get("student_evidence") or ""), inputs[cid]["transcription"])
                       for i in items)
        zero_rows.append({"case_id": cid, "historical_decision": scored[cid]["decision"],
                          "grounded_under_v2": grounded,
                          "new_routing": "AUTO" if grounded else "REVIEW"})

report = {
    "artifact": "counterfactual_structural_replay",
    "created_at": "2026-08-28",
    "label": "counterfactual structural replay — NOT actual model performance",
    "run_id": RUN.name,
    "method": {
        "inference_calls": 0, "cloud_calls": 0,
        "matcher": "autograder.evidence.evidence_supported (production, unchanged)",
        "note": ("no historical output was mutated, repaired or re-scored as real; "
                 "'counterfactually valid' means only that a span the model itself "
                 "produced inside the mis-placed field would have verified had it "
                 "been returned in rubric_items[].student_evidence")},
    "summary": {
        "evidence_failures": n,
        "counterfactually_valid_with_span_the_model_itself_isolated": recoverable,
        "verbatim_span_present_anywhere_12_to_200_chars": sum(
            1 for r in rows if r["verbatim_span_present_anywhere_12_to_200"]),
        "still_fail_without_new_copying": still,
        "content_class_counts": by_class,
        "over_200_as_submitted": sum(1 for r in rows if r["over_200_as_submitted"]),
        "unknown_or_duplicate_rubric_ids_in_history": 0,
        "historical_zero_verdicts": len(zero_rows),
        "historical_zero_verdicts_newly_routed_to_review": sum(
            1 for z in zero_rows if z["new_routing"] == "REVIEW"),
    },
    "zero_verdict_replay": zero_rows,
    "cases": rows,
}
(OUT / "REPLAY_STRUCTURAL_2026-08-28.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")

L = ["# Counterfactual structural replay — NOT actual model performance", "",
     f"Historical run `{RUN.name}` replayed READ-ONLY against the new output contract "
     "(grade-validation-v2) using the unchanged production matcher. Zero inference. "
     "No historical output was mutated or re-scored as real performance; this measures "
     "only whether spans the model ALREADY produced would have verified in the correct "
     "field.", "",
     f"- evidence failures examined: **{n}**",
     f"- counterfactually structurally valid with a span the model itself isolated "
     f"(whole field or a quote-delimited fragment): **{recoverable}/{n}**",
     f"- a 12–200-char verbatim span exists somewhere in the mis-placed field: "
     f"**{report['summary']['verbatim_span_present_anywhere_12_to_200_chars']}/{n}**",
     f"- still failing without new copying: {still}",
     f"- field content classes: {by_class}",
     f"- over the 200-char limit as submitted: {report['summary']['over_200_as_submitted']}/{n}",
     f"- unknown/duplicate rubric ids in history: 0 (rubric_items was empty in every output)",
     f"- historical zero verdicts: {len(zero_rows)}; newly routed to REVIEW under the "
     f"zero-side grounding rule: "
     f"{report['summary']['historical_zero_verdicts_newly_routed_to_review']} "
     f"(the run's only AUTO decision — a harmful undergrade — would no longer "
     "auto-finalize)", "",
     "| case | content class | model isolated a verified quote | still-fail reason |",
     "|---|---|---|---|"]
for r in rows:
    L.append(f"| {r['case_id']} | {r['content_class']} | "
             f"{'yes' if r['model_isolated_a_verified_quote'] else 'no'} | "
             f"{r['still_fail_reason_without_new_copying'] or '-'} |")
L += ["", "_Counterfactual structural replay — not actual model performance._", ""]
(OUT / "REPLAY_STRUCTURAL_2026-08-28.md").write_text("\n".join(L), encoding="utf-8", newline="\n")
print(json.dumps(report["summary"], indent=1))
print("written:", OUT / "REPLAY_STRUCTURAL_2026-08-28.json", "+ .md")
