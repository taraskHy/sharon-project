"""Recompute CALIBRATION metrics under the human-audited case status.

Reads ONLY persisted model outputs and the completed blinded audit. Makes no
provider call and rewrites no label — the audit changes which cases COUNT, not
what the ground truth is.

Status rules (from the audit taxonomy):

    A  keep the current benchmark label; the case counts normally
    B  keep the AUTHORITATIVE instructor label for alignment metrics; the case
       still counts, and a rubric-to-practice mismatch is recorded. The label is
       not changed and grade-v4 is not tuned on it.
    C  the transcription/evidence/rubric artifact is faulty, so the previous
       model comparison for that case is INVALID: drop it from the denominator
       and say so.
    D  genuinely ambiguous: excluded from STRICT accuracy, reported separately.

Both versions are always produced — pre-audit (every case counts) and revised —
because an audit that silently shrinks a denominator is indistinguishable from
one that improves a score.

    python scripts/calibration_audit_recompute.py
    python scripts/calibration_audit_recompute.py --audit-file PATH --json
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "evaluation" / "model_selection" / "runs" / "grade_primary"
DEFAULT_AUDIT = RUNS / "CALIBRATION_AUDIT_2026-08-26.json"

#: the four A/B arms of the completed v3-vs-v4 experiment
ARMS = {
    "A1": ("google/gemini-3.7-flash", "grade-v3",
           "calibration__calibration_verdict__all__google-gemini-3.7-flash__0868fe8923"),
    "A2": ("google/gemini-3.7-flash", "grade-v4-charitable",
           "calibration__calibration_verdict_v4__all__google-gemini-3.7-flash__9d4e07e8dc"),
    "B1": ("anthropic/claude-sonnet-5", "grade-v3",
           "calibration__calibration_verdict__all__anthropic-claude-sonnet-5__a163501062"),
    "B2": ("anthropic/claude-sonnet-5", "grade-v4-charitable",
           "calibration__calibration_verdict_v4__all__anthropic-claude-sonnet-5__a410eb2419"),
}
CLASSES = ("valid", "partially_valid")
#: statuses that remove a case from the STRICT accuracy denominator
EXCLUDING = {"C", "D"}


def case_status(audit: dict) -> dict[str, str | None]:
    """case_id -> human decision (None where the audit has not decided yet)."""
    return {c["case_id"]: c.get("human_decision") for c in audit.get("cases", [])}


def _metrics(pairs: list[tuple[str, str]]) -> dict:
    """pairs of (truth, predicted) -> the standard metric block."""
    from autograder.benchmark.roles import VERDICT_RANK

    n = len(pairs)
    if not n:
        return {"n": 0, "accuracy_pct": None, "macro_f1": None,
                "balanced_accuracy": None, "harmful_upgrades": 0,
                "harmful_downgrades": 0, "per_class": {}, "confusion": {}}
    conf = collections.Counter(pairs)
    correct = sum(v for (t, p), v in conf.items() if t == p)
    per, f1s, recs = {}, [], []
    for c in CLASSES:
        sup = sum(v for (t, _), v in conf.items() if t == c)
        tp = conf[(c, c)]
        prd = sum(v for (_, p), v in conf.items() if p == c)
        rec = tp / sup if sup else None
        pre = tp / prd if prd else None
        f1 = (2 * pre * rec / (pre + rec)) if (pre and rec) else 0.0
        per[c] = {"support": sup, "predicted": prd,
                  "precision": round(pre, 4) if pre is not None else None,
                  "recall": round(rec, 4) if rec is not None else None,
                  "f1": round(f1, 4)}
        if sup:
            f1s.append(f1)
            recs.append(rec or 0.0)
    return {
        "n": n,
        "accuracy_pct": round(100 * correct / n, 2),
        "macro_f1": round(statistics.mean(f1s), 4) if f1s else None,
        "balanced_accuracy": round(statistics.mean(recs), 4) if recs else None,
        "harmful_upgrades": sum(v for (t, p), v in conf.items()
                                if VERDICT_RANK[p] > VERDICT_RANK[t]),
        "harmful_downgrades": sum(v for (t, p), v in conf.items()
                                  if VERDICT_RANK[p] < VERDICT_RANK[t]),
        "per_class": per,
        "confusion": {f"{t}->{p}": v for (t, p), v in sorted(conf.items())},
    }


def recompute(audit_path: Path = DEFAULT_AUDIT) -> dict:
    from autograder.benchmark.manifests import load_manifest
    from autograder.benchmark.subsets import load_subset
    from autograder.benchmark.verdicts import verdict_from_model_score

    audit = json.loads(Path(audit_path).read_text(encoding="utf-8"))
    status = case_status(audit)
    m = load_manifest("grade_primary")
    by = {c.case_id: c for c in m.cases}
    frozen = [c["case_id"] for c in
              load_subset("grade_primary", "calibration_verdict_v4", m)["cases"]]

    excluded = sorted(cid for cid, s in status.items() if s in EXCLUDING)
    counts = collections.Counter(s for s in status.values() if s)
    undecided = sorted(cid for cid, s in status.items() if not s)

    out = {"audit_file": str(audit_path), "provider_calls": 0,
           "decisions": status, "decision_counts": dict(counts),
           "undecided": undecided,
           "excluded_from_strict_accuracy": excluded,
           "strict_denominator": len(frozen) - len(excluded),
           "note": ("Pre-audit metrics count all 12 cases. Revised metrics drop only cases the "
                    "human marked C (faulty artifact) or D (ambiguous). No label was rewritten; "
                    "B keeps the authoritative instructor label and is recorded as a "
                    "rubric-to-practice mismatch."),
           "rubric_to_practice_mismatches": sorted(c for c, s in status.items() if s == "B"),
           "arms": {}}

    for arm, (slug, pv, d) in ARMS.items():
        rows = {r["case_id"]: r for r in
                (json.loads(l) for l in (RUNS / d / "outputs.jsonl").read_text(
                    encoding="utf-8").splitlines() if l.strip())
                if r.get("ok") is not None}
        pre, rev = [], []
        for cid in frozen:
            r = rows.get(cid)
            if not r or not r.get("ok"):
                continue
            t = by[cid].label["explanation_verdict"]
            p = verdict_from_model_score(r["output"]["score"],
                                         by[cid].inputs["pack"]["max_score"])
            pre.append((t, p))
            if status.get(cid) not in EXCLUDING:
                rev.append((t, p))
        out["arms"][arm] = {"candidate": slug, "prompt_version": pv,
                            "pre_audit": _metrics(pre), "revised": _metrics(rev)}
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--audit-file", default=str(DEFAULT_AUDIT))
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    res = recompute(Path(a.audit_file))
    if a.json:
        print(json.dumps(res, ensure_ascii=False, indent=1))
        return
    print(f"decisions: {res['decision_counts'] or '(none recorded yet)'}")
    if res["undecided"]:
        print(f"UNDECIDED ({len(res['undecided'])}): {', '.join(res['undecided'])}")
    print(f"excluded from strict accuracy: {res['excluded_from_strict_accuracy'] or 'none'}")
    print(f"strict denominator: {res['strict_denominator']} of 12")
    if res["rubric_to_practice_mismatches"]:
        print(f"rubric-to-practice mismatches (B): {res['rubric_to_practice_mismatches']}")
    print()
    hdr = f"{'arm':4s} {'model':10s} {'prompt':9s} | {'pre n':>5s} {'acc%':>6s} {'mF1':>6s} " \
          f"{'balAcc':>7s} | {'rev n':>5s} {'acc%':>6s} {'mF1':>6s} {'balAcc':>7s}"
    print(hdr)
    print("-" * len(hdr))
    for arm, r in res["arms"].items():
        a_, b_ = r["pre_audit"], r["revised"]
        f = lambda v, w=6, p=2: (f"{v:{w}.{p}f}" if isinstance(v, (int, float)) else f"{'-':>{w}}")
        print(f"{arm:4s} {r['candidate'].split('/')[1][:10]:10s} {r['prompt_version'][6:15]:9s} | "
              f"{a_['n']:>5d} {f(a_['accuracy_pct'])} {f(a_['macro_f1'],6,4)} {f(a_['balanced_accuracy'],7,4)} | "
              f"{b_['n']:>5d} {f(b_['accuracy_pct'])} {f(b_['macro_f1'],6,4)} {f(b_['balanced_accuracy'],7,4)}")


if __name__ == "__main__":
    main()
