"""Zero-cost policy replay over an already-audited graded batch.

Re-runs the DETERMINISTIC grading stage (grade.grade_exam) on each exam's
persisted extraction.json with grading policies installed, using a
MockBackend that RAISES if any model call is attempted, and compares every
final total (and per-item points) against the frozen result.json.

Also simulates the explanation-bearing counterpart: for every MC row,
what the pre-OCR policy gate would have decided under wrong_choice_zero /
explanation_required_if_correct — counting explanation OCR calls, grader
calls, RAG lookups and REVIEW cases that would be avoided.

No OpenRouter calls, no local model calls, no network.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from autograder import grade  # noqa: E402
from autograder.backends import BackendConfig, BackendError  # noqa: E402
from autograder.backends.mock import MockBackend  # noqa: E402
from autograder.config import GraderConfig  # noqa: E402
from autograder.grade import VersionDecision  # noqa: E402
from autograder.policies import MCResolution, decide_before_ocr  # noqa: E402
from autograder.schema import AnswerKey, ExamExtraction  # noqa: E402


class NoCallBackend(MockBackend):
    def parse(self, **kw):
        raise BackendError("policy replay must make ZERO model calls")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--policy", default="wrong_choice_zero",
                    help="policy applied to every question for the replay")
    ap.add_argument("--min-confidence", type=float, default=0.9)
    args = ap.parse_args()

    job = REPO / args.job if not Path(args.job).is_absolute() else Path(args.job)
    key = AnswerKey.model_validate_json((job / "uploads" / "answer_key.json").read_text(encoding="utf-8"))
    policies = {q.id: args.policy for q in key.questions}
    llm = NoCallBackend(config=BackendConfig(backend="mock", model="none"))

    totals_identical = True
    n_exams = n_q = n_items = 0
    items_identical = 0
    early = Counter()
    sim = Counter()
    per_exam = []
    for exam_dir in sorted((job / "exams").glob("exam-*")):
        rp, ep = exam_dir / "result.json", exam_dir / "extraction.json"
        if not (rp.exists() and ep.exists()):
            continue
        ref = json.loads(rp.read_text(encoding="utf-8"))
        ext = ExamExtraction.model_validate_json(ep.read_text(encoding="utf-8"))
        version = ref.get("detected_version") or ref.get("variant_detection", {}).get("selected_variant") or key.versions[0]
        vdec = VersionDecision(version=version, description="replay: frozen version", uncertain=False)
        grade.set_grading_policies(policies, args.min_confidence)
        try:
            # judge_all with the no-call backend: the policy gate must settle
            # everything that would otherwise reach the judge (MC-only key ->
            # nothing needs judging anyway; the backend raises if touched)
            judgements = grade.judge_all(llm, key, ext, version)
            res = grade.grade_exam(key, ext, judgements, vdec, GraderConfig(version=version))
        finally:
            log = grade.early_exit_log()
            grade.set_grading_policies(None)
        n_exams += 1
        same = abs(float(res.total_awarded) - float(ref["total_awarded"])) < 1e-9
        totals_identical &= same
        # per-item comparison
        ref_items = {(q["question_id"], s["sub_item_id"]): s for q in ref["questions"] for s in q["sub_results"]}
        for q in res.questions:
            n_q += 1
            for s in q.sub_results:
                n_items += 1
                r = ref_items.get((q.question_id, s.sub_item_id))
                if r and abs(float(s.points_total) - float(r["points_total"])) < 1e-9:
                    items_identical += 1
        for e in log:
            early[e["flag"]] += 1
        # simulate the explanation-bearing counterpart per MC row
        for q in key.questions:
            eq = ext.question(q.id)
            for se in eq.sub_items:
                sel = grade.normalize_answer(se.final_answer)
                state = ("single_mark" if se.status == "answered" and sel else "blank" if se.status == "unanswered"
                         else "multiple_marks" if se.status == "ambiguous" else "unclear")
                mc = MCResolution(sel, state, float(se.confidence or 0.0), "deterministic", list(se.candidate_answers or []))
                ks = next(s for s in q.sub_items if s.id == se.sub_item_id)
                acc = grade._accepted(ks, version)
                for pol in ("wrong_choice_zero", "explanation_required_if_correct", "choice_only"):
                    d = decide_before_ocr(policy=pol, mc=mc, accepted=acc, points_selection=float(ks.points),
                                          points_max=float(ks.points), min_confidence=args.min_confidence)
                    sim[(pol, d.action, bool(d.skip_explanation))] += 1
        per_exam.append({"exam": exam_dir.name, "replay_total": res.total_awarded, "ref_total": ref["total_awarded"],
                         "identical": same, "review_items": len(res.needs_human_review),
                         "ref_review_items": len(ref.get("needs_human_review", []))})

    print(f"exams: {n_exams} | questions: {n_q} | MC rows: {n_items}")
    print(f"per-item points identical: {items_identical}/{n_items} | all totals identical: {totals_identical}")
    for pe in per_exam:
        print(f"  {pe['exam']}: replay {pe['replay_total']} vs ref {pe['ref_total']} "
              f"{'OK' if pe['identical'] else 'DIFF'} | review {pe['review_items']} (ref {pe['ref_review_items']})")
    print(f"deterministic early exits recorded (policy={args.policy}): {dict(early)}")
    print("\nsimulated explanation-bearing counterpart (per MC row, per policy):")
    for pol in ("wrong_choice_zero", "explanation_required_if_correct", "choice_only"):
        skip = sum(v for (p, a, s), v in sim.items() if p == pol and s)
        ocr = sum(v for (p, a, s), v in sim.items() if p == pol and a == "ocr_explanation")
        rev = sum(v for (p, a, s), v in sim.items() if p == pol and a == "review")
        print(f"  {pol:34} rows={n_items} skip_explanation(no OCR/RAG/grader)={skip} "
              f"ocr_needed={ocr} review={rev}")
    return 0 if totals_identical else 1


if __name__ == "__main__":
    sys.exit(main())
