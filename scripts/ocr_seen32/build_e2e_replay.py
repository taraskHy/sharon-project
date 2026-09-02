"""Mission H: replay the production pipeline from PERSISTED artifacts only.

Never synthesises a grader output. The point is to find integration gaps, not
to manufacture an automation percentage. ZERO inference.
"""
import hashlib, json, re, subprocess, time
from collections import Counter
from pathlib import Path

from autograder.benchmark.manifests import load_manifest
from autograder.benchmark.ocr_fallback import replay
from autograder.benchmark.ocr_outcomes import classify_row

R = Path("evaluation/model_selection/runs/ocr_primary")
LG = Path("evaluation/model_selection/runs/local_grade_primary")
S32 = Path("evaluation/model_selection/runs_seen32/ocr_primary")
GEM, SON = "google/gemini-3.7-flash", "anthropic/claude-sonnet-5"
DIRS = {GEM: S32 / "dev__seen46_ocr_dev__all__google-gemini-3.7-flash__c4ae61f634",
        SON: S32 / "dev__seen46_ocr_dev__all__anthropic-claude-sonnet-5__2f3a7c346c"}
man = load_manifest("ocr_primary")
by = {c.case_id: c for c in man.cases}
base = json.loads((R / "OCR_SEEN32_BASELINE_2026-09-02.json").read_text(encoding="utf-8"))
ORDER = base["ordered_case_ids"]


def grading_case_id(crop_id: str) -> str:
    """hc_e002_q1_r1 -> e002_q1_r1 ; hl_e003_q1_r1__l1 -> e003_q1_r1"""
    s = re.sub(r"^(hc|hl)_", "", crop_id)
    return re.sub(r"__l\d+$", "", s)


rows_by_model, tax = {}, {}
for slug, d in DIRS.items():
    rows = {json.loads(l)["case_id"]: json.loads(l)
            for l in (d / "outputs.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    rows_by_model[slug] = rows
    tax[slug] = {c: classify_row(rows[c], by[c].label["reference"]) for c in ORDER}

gt = {c: (rows_by_model[GEM][c].get("output") or {}).get("transcription") for c in ORDER}
st = {c: (rows_by_model[SON][c].get("output") or {}).get("transcription") for c in ORDER}
fb = replay(ORDER, tax[GEM], tax[SON], gt, st, primary_model=GEM, secondary_model=SON)
fb_by_case = {d["case_id"]: d for d in fb["decisions"]}

# ---- persisted grader results ---------------------------------------------
grader = {}
for f in sorted(LG.glob("*.jsonl")):
    for line in f.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        cid = r.get("case_id")
        if cid:
            grader.setdefault(cid, []).append({"source": f.name, **r})

# ---- persisted risk decisions ---------------------------------------------
risk = {}
sp = LG / "SHADOW_REPLAY_2026-09-02.jsonl"
if sp.exists():
    for line in sp.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        risk.setdefault(r["case_id"], []).append(r)

stages, per_case = Counter(), []
for cid in ORDER:
    gcid = grading_case_id(cid)
    d = fb_by_case[cid]
    has_ocr = d["resolved"]
    g_rows = grader.get(gcid, [])
    r_rows = risk.get(gcid, [])
    if not has_ocr:
        stage = "STOPS_AT_OCR_no_usable_transcription"
    elif not g_rows:
        stage = "STOPS_AT_OCR_no_persisted_grader_run"
    elif not r_rows:
        stage = "STOPS_AT_GRADER_no_persisted_risk_decision"
    else:
        stage = "FULL_E2E_EVIDENCE"
    stages[stage] += 1
    decisions = sorted({x["decision"]["action"] for x in r_rows if isinstance(x.get("decision"), dict)})
    per_case.append({
        "crop_id": cid, "grading_case_id": gcid,
        "ocr_resolved": has_ocr, "ocr_source_model": d["chosen_model"],
        "ocr_fallback_used": d["fallback_used"], "ocr_needs_review": d["needs_review"],
        "persisted_grader_runs": len(g_rows),
        "persisted_grader_sources": sorted({x["source"] for x in g_rows}),
        "persisted_risk_decisions": len(r_rows),
        "risk_actions": decisions,
        "stage_reached": stage,
    })

# ---- what the gap actually is ---------------------------------------------
gap_note = (
    "The 32 OCR crops and the persisted grader runs address DIFFERENT units. A crop is one "
    "handwritten line or answer cell; a grader case is a whole question response. Several crops map "
    "to the same grading case id, and the persisted grader runs were produced from the frozen "
    "hebrew_bench_v2 evidence, NOT from these new OCR transcriptions. So a 'FULL_E2E_EVIDENCE' row "
    "means the persisted artifacts exist end to end for that unit — it does NOT mean this OCR output "
    "was fed to that grader run. Nothing here was re-graded, and no grader output was synthesised.")

art = {
    "artifact": "ocr_seen32_e2e_persisted_replay",
    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    "provider_calls": 0, "grader_calls": 0,
    "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
    "purpose": "identify integration gaps between OCR, the local grader and the risk engine",
    "method": ("join the 32 OCR crops to persisted grader/risk artifacts by derived grading case id; "
               "never synthesise a grader output where none exists"),
    "unit_mismatch_warning": gap_note,
    "stage_counts": dict(stages),
    "distinct_grading_cases_touched": len({p["grading_case_id"] for p in per_case}),
    "crops_per_grading_case": dict(Counter(Counter(p["grading_case_id"] for p in per_case).values())),
    "ocr_coverage": {"resolved": fb["primary_used"] + fb["fallback_used"],
                     "unresolved": fb["unresolved"],
                     "needs_human_review": fb["needs_review"]},
    "integration_gaps": [
        "OCR crops and grader cases are different units; there is no persisted mapping table that "
        "says which crops compose a grading case's evidence. The join here is derived from the id "
        "convention and is therefore a convention, not a contract.",
        "No persisted grader run consumes these new OCR transcriptions — the grading corpus was "
        "built from the frozen hebrew_bench_v2 evidence. An end-to-end run on NEW OCR output has "
        "never been executed.",
        "The risk engine has persisted decisions only for the shadow replay population, which is "
        "the seen-46 grading set, not this crop set.",
    ],
    "per_case": per_case,
}
body = json.dumps(art, ensure_ascii=False, indent=1)
art["content_sha256"] = hashlib.sha256(body.encode()).hexdigest()
p = R / "OCR_SEEN32_E2E_REPLAY_2026-09-02.json"
p.write_text(json.dumps(art, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
print("wrote", p)
print("stage counts:", json.dumps(dict(stages), indent=1))
print("distinct grading cases touched:", art["distinct_grading_cases_touched"])
print("crops per grading case histogram:", art["crops_per_grading_case"])
print("ocr coverage:", json.dumps(art["ocr_coverage"]))
