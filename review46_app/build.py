"""Build the deterministic 46-case review bundle + the admin-only sources.

Reviewer-facing content = the blind labeling bundle (question, rubric,
official solution, frozen transcription, answer crops, red-masked source
page, opaque ids). Admin-only content = ``private/instructor_reference.json``
and ``private/model_proposals.json`` — the comparison sources, deliberately
NEVER part of any reviewer payload.

Hard guarantees, enforced here and by tests:
* exactly the frozen campaign's 46 cases, in a bundle whose id salt is the
  campaign hash (distinct from the owner-labeling bundle's ids);
* zero HELD_OUT / forbidden-writer content anywhere in the bundle tree;
* no machine-absolute path in any served payload;
* a fixed review_bundle_sha256 over every bundle file.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from . import CAMPAIGN  # noqa: E402

DATASET = REPO / "evaluation" / "model_selection" / "datasets" / "grade_primary"
CAMPAIGN_PATH = REPO / "evaluation" / "model_selection" / "experiments" / \
    "LOCAL_GRADE_PRIMARY_SEEN_46_CAMPAIGN_2026-08-28.json"
FORBIDDEN_WRITERS = ("e005", "e006")


def _sha_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _campaign() -> dict:
    doc = json.loads(CAMPAIGN_PATH.read_text(encoding="utf-8"))
    assert doc["campaign"] == CAMPAIGN
    return doc


def _model_proposals(campaign: dict, id_of: dict[str, str]) -> dict:
    """The frozen model outputs of the SEEN-46 run, keyed by OPAQUE item id.
    Immutable admin/adjudication data; never in a reviewer payload."""
    from scripts.seen46_campaign import find_campaign_runs
    runs = find_campaign_runs()
    assert set(runs) == {"DEV", "CALIBRATION"}, f"campaign runs incomplete: {sorted(runs)}"
    out: dict[str, dict] = {}
    for split, d in runs.items():
        run = json.loads((d / "run.json").read_text(encoding="utf-8"))
        scored = {r["case_id"]: r for r in json.loads((d / "scored.jsonl.json").read_text(encoding="utf-8"))}
        outputs = {json.loads(l)["case_id"]: json.loads(l) for l in (d / "outputs.jsonl").open(encoding="utf-8")}
        for cid, s in scored.items():
            o = outputs[cid]
            g = o.get("output") if o.get("ok") else None
            out[id_of[cid]] = {
                "case_id": cid,
                "run_id": run["run_id"],
                "model": run["config"]["candidate"],
                "prompt_version": run["config"]["prompt_version"],
                "prompt_sha256": run["config"]["prompt_sha256"],
                "schema_sha256": run["config"]["schema_sha256"],
                "raw_score": (g or {}).get("score"),
                "verdict": s.get("predicted_verdict"),
                "rubric_items": (g or {}).get("rubric_items"),
                "evidence_field": (g or {}).get("evidence"),
                "uncertain": (g or {}).get("uncertain"),
                "validation_ok": s.get("validation_ok"),
                "validation_problems": s.get("validation_problems"),
                "decision": s.get("decision"),
                "latency_s": o.get("latency_s"),
                "cache_hit": o.get("cache_hit"),
                "ts": o.get("ts"),
            }
    missing = sorted(set(id_of.values()) - set(out))
    assert not missing, f"model proposal missing for {len(missing)} item(s)"
    return out


def _instructor_reference(campaign: dict, id_of: dict[str, str]) -> dict:
    """The original instructor grades + campaign case metadata, keyed by
    opaque id. REFERENCE source: recorded, immutable, never a reviewer field."""
    finals = json.loads((DATASET / "final_labels.json").read_text(encoding="utf-8"))["labels"]
    out: dict[str, dict] = {}
    for c in campaign["cases"]:
        cid = c["case_id"]
        f = finals[cid]
        out[id_of[cid]] = {
            "case_id": cid,
            "actual_instructor_score": c["actual_instructor_score"],
            "ground_truth_source": f.get("ground_truth_source"),
            "finalized_at": f.get("finalized_at"),
            "selection_correct": c["selection_correct"],
            "selection_correct_source": c["selection_correct_source"],
            "instructor_derived_verdict": c["instructor_derived_verdict"],
            "verdict_derivable": c["verdict_derivable"],
            "verdict_derivation_reason": c["verdict_derivation_reason"],
            "evidence_issue_flag": c["evidence_issue_flag"],
            "audit_flag": c["audit_flag"],           # diagnostic metadata, never a target
            "strict_verdict_eligible": c["strict_verdict_eligible"],
            "split": c["split"],                      # admin-only context
        }
    return out


def build_review_bundle(out_dir: Path | None = None, *, replace: bool = False) -> dict:
    from labeling_app.bundle import Bundle, build_bundle
    from . import default_data_dir

    campaign = _campaign()
    case_ids = [c["case_id"] for c in campaign["cases"]]
    assert len(case_ids) == 46
    for cid in case_ids:
        assert cid.split("_")[0] not in FORBIDDEN_WRITERS, cid

    out = Path(out_dir) if out_dir else default_data_dir() / "bundle"
    meta = build_bundle(DATASET, out, evaluation_root=REPO / "evaluation",
                        salt=campaign["campaign_sha256"], case_ids=case_ids, replace=replace)
    bundle = Bundle(out)

    # every campaign case must have become a reviewable item (the labeling
    # eligibility gate must not silently drop any — reviewers judge the
    # EXPLANATION independently of the selection, so all 46 are reviewable)
    got = set(bundle.id_map.values())
    if got != set(case_ids):
        raise RuntimeError(f"bundle holds {len(got)} of 46 campaign cases; "
                           f"missing {sorted(set(case_ids) - got)[:5]} — refusing")
    id_of = {cid: oid for oid, cid in bundle.id_map.items()}

    # HELD_OUT / forbidden-writer scan over EVERY text file in the bundle
    # tree. Word-ish match: a 64-char sha256 hex string can contain "e006" by
    # chance, so a match INSIDE a longer hex run is not a writer id.
    import re
    forbidden_pat = re.compile(r"(?<![0-9a-fA-F])(" + "|".join(FORBIDDEN_WRITERS) + r")(?![0-9a-fA-F])")
    for p in sorted(out.rglob("*.json")):
        m = forbidden_pat.search(p.read_text(encoding="utf-8"))
        if m:
            raise RuntimeError(f"forbidden writer {m.group(1)} appears in {p}")
    # no machine-absolute paths in served payloads
    items_text = (out / "items.json").read_text(encoding="utf-8")
    for marker in (":\\\\", ":/Users", "C:/Users", "/home/"):
        if marker in items_text:
            raise RuntimeError(f"machine path marker {marker!r} in items.json")

    (out / "private").mkdir(exist_ok=True)
    proposals = _model_proposals(campaign, id_of)
    reference = _instructor_reference(campaign, id_of)
    (out / "private" / "model_proposals.json").write_text(
        json.dumps(proposals, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
    (out / "private" / "instructor_reference.json").write_text(
        json.dumps(reference, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
    (out / "private" / "campaign.json").write_text(json.dumps({
        "campaign": CAMPAIGN, "campaign_sha256": campaign["campaign_sha256"],
        "cases": len(case_ids), "dev": campaign["population"]["dev"],
        "calibration": campaign["population"]["calibration"], "held_out": 0,
        "wanted_reviews_per_case": 2,
    }, indent=1, sort_keys=True), encoding="utf-8")

    # fixed content hash over every file in the bundle (deterministic order)
    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "bundle46.json")
    h = hashlib.sha256()
    for p in files:
        h.update(str(p.relative_to(out)).replace("\\", "/").encode())
        h.update(_sha_file(p).encode())
    doc = {"campaign": CAMPAIGN, "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
           "cases": len(case_ids), "files": len(files),
           "review_bundle_sha256": h.hexdigest(),
           "campaign_sha256": campaign["campaign_sha256"],
           "labeling_bundle_meta": {k: meta.get(k) for k in ("items", "items_sha256", "pages_rendered")
                                    if k in meta}}
    (out / "bundle46.json").write_text(json.dumps(doc, indent=1, sort_keys=True), encoding="utf-8")
    return doc


__all__ = ["build_review_bundle", "CAMPAIGN_PATH", "DATASET", "FORBIDDEN_WRITERS"]
