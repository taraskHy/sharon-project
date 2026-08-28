"""Per-role benchmark adapters.

An adapter turns one BenchCase into the exact request production would send
(same system prompt, same structured-output schema, same block layout), then
scores the structured output against the case's evaluation-side label and
aggregates per-run metrics. Adapters see ``case.inputs`` when building a
request and ``case.label`` ONLY when scoring — ``build_request`` receives the
inputs dict alone, so a leak would have to be written on purpose (and the
runner's leakage_check would still catch it).

Prompt/schema provenance is part of every run record: ``prompt_version``,
sha256 of the system prompt text, sha256 of the output JSON schema, and
``adapter_version`` (bump when scoring semantics change).
"""
from __future__ import annotations

import base64
import hashlib
import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel

from .manifests import BenchCase, REPO_ROOT


@dataclass
class Request:
    system: str
    content_blocks: list[dict]
    output_model: type[BaseModel]
    prompt_version: str
    max_tokens: int | None = None

    def text_for_inspection(self) -> str:
        """Everything textual the model sees (images as placeholders)."""
        parts = [self.system]
        for b in self.content_blocks:
            if b.get("type") == "text":
                parts.append(str(b.get("text", "")))
            elif b.get("type") == "image":
                src = b.get("source") or {}
                parts.append(f"<image {src.get('media_type', '?')} {len(str(src.get('data', '')))} b64 chars>")
        return "\n".join(parts)

    def provenance(self) -> dict[str, str]:
        schema = json.dumps(self.output_model.model_json_schema(), sort_keys=True)
        return {"prompt_version": self.prompt_version,
                "prompt_sha256": hashlib.sha256(self.system.encode("utf-8")).hexdigest(),
                "schema_name": self.output_model.__name__,
                "schema_sha256": hashlib.sha256(schema.encode("utf-8")).hexdigest()}


def _image_block_from_file(bench_root: Path, rel: str) -> dict:
    p = (Path(bench_root) / rel) if not Path(rel).is_absolute() else Path(rel)
    data = base64.standard_b64encode(p.read_bytes()).decode("ascii")
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}}


def _pct(n: int, d: int) -> float | None:
    return round(100.0 * n / d, 2) if d else None


#: ordering of the canonical verdicts, low to high credit
VERDICT_RANK = {"invalid": 0, "partially_valid": 1, "valid": 2}


def _verdict_metrics(judged: list[dict], classes: tuple[str, ...]) -> dict[str, Any]:
    """Classification metrics for the canonical explanation verdict.

    Accuracy alone hides the rare classes (``partially_valid`` is 4 of 32 DEV
    cases), so the confusion matrix and macro-F1 are reported alongside it and
    per-class precision/recall is reported only where the class actually has
    support — an F1 over an empty class is noise, not a measurement.
    """
    n = len(judged)
    correct = sum(1 for r in judged if r["verdict_exact"])
    # confusion[truth][predicted]
    confusion = {t: {p: 0 for p in classes} for t in classes}
    for r in judged:
        t, p = r["label_verdict"], r["predicted_verdict"]
        confusion.setdefault(t, {}).setdefault(p, 0)
        confusion[t][p] += 1

    per_class: dict[str, Any] = {}
    f1s = []
    recalls_full: list[float] = []           # FULL precision — rounding before
    for c in classes:                        # averaging overstated balanced
        tp = confusion.get(c, {}).get(c, 0)  # accuracy by 1e-4 (review 2026-08-28)
        support = sum(confusion.get(c, {}).values())
        predicted = sum(confusion.get(t, {}).get(c, 0) for t in confusion)
        if support == 0 and predicted == 0:
            per_class[c] = {"support": 0, "precision": None, "recall": None, "f1": None,
                            "note": "no support and never predicted — not measured"}
            continue
        precision = tp / predicted if predicted else None
        recall = tp / support if support else None
        if precision and recall:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0 if support or predicted else None
        per_class[c] = {"support": support, "predicted": predicted,
                        "precision": round(precision, 4) if precision is not None else None,
                        "recall": round(recall, 4) if recall is not None else None,
                        "f1": round(f1, 4) if f1 is not None else None}
        if support:
            f1s.append(f1 or 0.0)
            recalls_full.append(recall or 0.0)
    return {
        "verdict_exact_pct": _pct(correct, n),
        "verdict_balanced_accuracy": (round(statistics.mean(recalls_full), 4)
                                      if recalls_full else None),
        "harmful_verdict_upgrades": sum(1 for r in judged if r.get("harmful_verdict_upgrade")),
        "harmful_verdict_downgrades": sum(1 for r in judged if r.get("harmful_verdict_downgrade")),
        "verdict_confusion": confusion,
        "verdict_per_class": per_class,
        "verdict_macro_f1": round(statistics.mean(f1s), 4) if f1s else None,
        "verdict_classes_with_support": sum(1 for c in classes
                                            if sum(confusion.get(c, {}).values())),
    }


def _usage_stats(rows: list[dict]) -> dict[str, Any]:
    live = [r for r in rows if r.get("ok") is not None and not r.get("cache_hit")]
    lat = [float(r["latency_s"]) for r in rows if r.get("latency_s") is not None and not r.get("cache_hit")]
    def _sum(k):
        return sum(int((r.get("usage") or {}).get(k) or 0) for r in live)
    return {
        "calls_live": len(live),
        "cache_hits": sum(1 for r in rows if r.get("cache_hit")),
        "latency_mean_s": round(statistics.mean(lat), 3) if lat else None,
        "latency_median_s": round(statistics.median(lat), 3) if lat else None,
        "input_tokens": _sum("input_tokens"), "output_tokens": _sum("output_tokens"),
        "reasoning_tokens": _sum("reasoning_tokens"), "total_tokens": _sum("total_tokens"),
        "reported_cost": round(sum(float((r.get("usage") or {}).get("reported_cost") or 0) for r in live), 6),
    }


# ----------------------------------------------------------------------------
# text metrics (canonical: scripts/hebrew_bench_eval.py via refaudit)
# ----------------------------------------------------------------------------

class _TextMetrics:
    """normalize / lev from scripts/hebrew_bench_eval.py (the frozen
    evaluator's canonical definitions, loaded via refaudit._load_metric_fns)
    plus refaudit.digit_op_signature — one namespace for the adapters."""

    def __init__(self):
        from .manifests import _load_refaudit
        ra = _load_refaudit()
        self.normalize, self.lev, self.word_align = ra._load_metric_fns()
        self.digit_op_signature = ra.digit_op_signature


_TM: _TextMetrics | None = None


def _textmetrics() -> _TextMetrics:
    global _TM
    if _TM is None:
        _TM = _TextMetrics()
    return _TM


# ----------------------------------------------------------------------------
# OCR_VERIFY (B2)
# ----------------------------------------------------------------------------

class OcrVerifyAdapter:
    role = "ocr_verify"
    task = "ocr_verify"
    adapter_version = "ocr-verify-bench-v1"
    prompt_version = "ocr-verify-v1"
    #: the ONLY input fields an ocr_verify request may be built from
    model_visible_fields = ("case_id", "crop", "candidate_transcription")
    default_max_tokens = 400

    def build_request(self, inputs: dict, bench_root: Path) -> Request:
        from ..escalation import OCR_VERIFY_SYSTEM, OCRVerifyResult
        # EXACTLY production's block layout (escalation.escalate_ocr)
        return Request(
            system=OCR_VERIFY_SYSTEM,
            content_blocks=[
                _image_block_from_file(bench_root, inputs["crop"]),
                {"type": "text", "text": "Proposed transcription:\n" + inputs["candidate_transcription"]
                                         + "\nCheck fidelity now."},
            ],
            output_model=OCRVerifyResult, prompt_version=self.prompt_version,
            max_tokens=self.default_max_tokens)

    @staticmethod
    def accepted(out: dict | None) -> bool | None:
        """Production's AUTO gate (escalation.escalate_ocr): supported AND
        high/medium confidence AND no reported omissions/substitutions/additions."""
        if not out:
            return None
        return bool(out.get("verdict") == "supported"
                    and out.get("confidence") in ("high", "medium")
                    and not (out.get("omissions") or out.get("substitutions") or out.get("additions")))

    def score(self, case: BenchCase, output: dict | None, error: str | None) -> dict:
        acc = self.accepted(output)
        positive = case.label.get("polarity") == "positive"
        row = {"case_id": case.case_id, "component": case.component, "split": case.split,
               "polarity": case.label.get("polarity"), "expected_verdict": case.label.get("expected_verdict"),
               "verdict": (output or {}).get("verdict"), "confidence": (output or {}).get("confidence"),
               "accepted": acc, "schema_failure": output is None,
               "corruption_type": case.label.get("corruption_type"),
               "synthetic_group": case.label.get("synthetic_group"),
               "error_kinds": case.label.get("error_kinds")}
        if acc is None:
            row["outcome"] = "schema_failure"
        elif positive:
            row["outcome"] = "true_accept" if acc else "false_reject"
        else:
            row["outcome"] = "false_accept" if acc else "true_reject"
        return row

    @staticmethod
    def _rates(rows: list[dict]) -> dict:
        neg = [r for r in rows if r["polarity"] == "negative"]
        pos = [r for r in rows if r["polarity"] == "positive"]
        fa = sum(1 for r in neg if r["outcome"] == "false_accept")
        fr = sum(1 for r in pos if r["outcome"] == "false_reject")
        ta = sum(1 for r in pos if r["outcome"] == "true_accept")
        schema = sum(1 for r in rows if r["outcome"] == "schema_failure")
        reviews = sum(1 for r in rows if r["accepted"] is False)
        return {
            "cases": len(rows), "positives": len(pos), "negatives": len(neg),
            "false_accepts": fa, "false_accept_rate_pct": _pct(fa, len(neg)),
            "false_rejects": fr, "false_reject_rate_pct": _pct(fr, len(pos)),
            "supported_precision_pct": _pct(ta, ta + fa) if (ta + fa) else None,
            "review_rate_pct": _pct(reviews, len(rows)),
            "schema_failures": schema, "schema_failure_rate_pct": _pct(schema, len(rows)),
        }

    def aggregate(self, scored: list[dict], raw_rows: list[dict]) -> dict:
        real = [r for r in scored if r["component"] == "REAL"]
        synth = [r for r in scored if r["component"] == "SYNTHETIC"]
        by_type: dict[str, dict] = {}
        for t in sorted({r.get("corruption_type") for r in synth if r.get("corruption_type")}):
            by_type[t] = self._rates([r for r in synth if r.get("corruption_type") == t])
        numeric = [r for r in synth if r.get("synthetic_group") == "numeric_math"]
        out = {
            "primary_metric": "false_accept_rate_pct (REAL); report SYNTHETIC separately; COMBINED secondary",
            "REAL": self._rates(real) if real else None,
            "SYNTHETIC": ({**self._rates(synth), "by_corruption_type": by_type,
                           "numeric_math": self._rates(numeric) if numeric else None} if synth else None),
            "COMBINED_secondary": self._rates(scored) if (real and synth) else None,
            "usage": _usage_stats(raw_rows),
        }
        return out


# ----------------------------------------------------------------------------
# OCR_PRIMARY (B1)
# ----------------------------------------------------------------------------

class BenchTranscription(BaseModel):
    """History-compatible output envelope of the hebrew_bench_v2 runs
    (scripts/m2_bench_run.py SCHEMA): {"transcription": str}."""

    transcription: str


#: The historical hebrew_bench_v2 prompts (scripts/m2_bench_run.py PROMPTS),
#: kept byte-identical so new runs stay comparable with historical outputs
#: (tests/test_benchmark_harness.py asserts parity against the script).
OCR_PRIMARY_PROMPTS: dict[str, str] = {}


def _load_historical_prompts() -> dict[str, str]:
    """Parse PROMPTS/STRICT_RULES out of scripts/m2_bench_run.py WITHOUT
    importing it (it pulls cv2/numpy/httpx adapters)."""
    global OCR_PRIMARY_PROMPTS
    if OCR_PRIMARY_PROMPTS:
        return OCR_PRIMARY_PROMPTS
    import ast
    src = (REPO_ROOT / "scripts" / "m2_bench_run.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    ns: dict[str, Any] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in ("STRICT_RULES", "PROMPTS"):
                code = ast.Module(body=[node], type_ignores=[])
                exec(compile(code, "m2_bench_run.PROMPTS", "exec"), ns)  # noqa: S102 — literals only
    prompts = ns.get("PROMPTS")
    if not isinstance(prompts, dict):
        raise RuntimeError("could not recover PROMPTS from scripts/m2_bench_run.py")
    OCR_PRIMARY_PROMPTS = {str(k): str(v) for k, v in prompts.items()}
    return OCR_PRIMARY_PROMPTS


class OcrPrimaryAdapter:
    role = "ocr_primary"
    task = "ocr_primary"
    adapter_version = "ocr-primary-bench-v1"
    prompt_version = "m2-strict-v1"
    model_visible_fields = ("case_id", "image", "category", "task")
    default_max_tokens = 600

    def build_request(self, inputs: dict, bench_root: Path) -> Request:
        prompts = _load_historical_prompts()
        cat = inputs["category"]
        if cat not in prompts:
            raise KeyError(f"no historical prompt for category {cat!r}")
        return Request(system=prompts[cat],
                       content_blocks=[_image_block_from_file(bench_root, inputs["image"])],
                       output_model=BenchTranscription, prompt_version=self.prompt_version,
                       max_tokens=self.default_max_tokens)

    def score(self, case: BenchCase, output: dict | None, error: str | None) -> dict:
        ra = _textmetrics()
        ref = case.label.get("reference")
        hyp = (output or {}).get("transcription")
        row = {"case_id": case.case_id, "split": case.split, "category": case.meta.get("category"),
               "tier": case.meta.get("tier"), "hard": case.label.get("hard"),
               "reference_status": case.label.get("reference_status"),
               "schema_failure": output is None, "cer": None, "usable_25": None, "usable_50": None,
               "number_sign_error": None, "scored": False}
        if not case.label.get("provenance_valid"):
            # never score against an inadmissible reference (no silent fallback)
            row["skip_reason"] = f"invalid_reference_source:{case.label.get('provenance_class')}"
            return row
        if ref is None or hyp is None:
            row["skip_reason"] = "no_reference" if ref is None else "schema_failure"
            return row
        g, h = ra.normalize(ref), ra.normalize(hyp)
        cer = (ra.lev(g, h) / len(g)) if g else (0.0 if not h else 1.0)
        row.update({"cer": round(cer, 4), "usable_25": cer <= 0.25, "usable_50": cer <= 0.50,
                    "number_sign_error": ra.digit_op_signature(ref) != ra.digit_op_signature(hyp),
                    "scored": True})
        return row

    def aggregate(self, scored: list[dict], raw_rows: list[dict]) -> dict:
        ok = [r for r in scored if r["scored"]]
        cers = [r["cer"] for r in ok]
        def _block(rows):
            cs = [r["cer"] for r in rows]
            return {"cases": len(rows), "mean_cer": round(statistics.mean(cs), 4) if cs else None,
                    "median_cer": round(statistics.median(cs), 4) if cs else None,
                    "usable_le_0.25_pct": _pct(sum(1 for r in rows if r["usable_25"]), len(rows)),
                    "usable_le_0.50_pct": _pct(sum(1 for r in rows if r["usable_50"]), len(rows)),
                    "number_sign_formula_errors": sum(1 for r in rows if r["number_sign_error"])}
        by_cat = {c: _block([r for r in ok if r["category"] == c])
                  for c in sorted({r["category"] for r in ok})}
        return {
            "primary_metric": "mean_cer (lower is better) on audited references; usable rates secondary",
            "overall": _block(ok),
            "by_category": by_cat,
            "hard_items": _block([r for r in ok if r.get("hard")]),
            "schema_failures": sum(1 for r in scored if r["schema_failure"]),
            "unscored_no_reference": sum(1 for r in scored if r.get("skip_reason") == "no_reference"),
            "refused_invalid_reference": sum(1 for r in scored if str(r.get("skip_reason", "")).startswith("invalid_reference_source")),
            "usage": _usage_stats(raw_rows),
        }


# ----------------------------------------------------------------------------
# GRADE_PRIMARY / GRADE_ESCALATE (B3 / B4)
# ----------------------------------------------------------------------------

def pack_from_inputs(p: dict):
    """Rebuild a QuestionGradingPack from the serialized, RAG-free pack in a
    benchmark case. Refuses packs that smuggle retrieval (NO-RAG benchmark)."""
    from ..gradingpack import QuestionGradingPack, RubricItemSpec
    if p.get("rag_evidence") or p.get("rag_prepared"):
        raise ValueError("grading benchmark packs must be NO-RAG (rag_evidence present)")
    items = [RubricItemSpec(**ri) for ri in p.get("rubric_items", [])]
    return QuestionGradingPack(
        question_id=p["question_id"], question_text=p["question_text"],
        question_type=p.get("question_type", "open"), max_score=float(p["max_score"]),
        correct_by_version=p.get("correct_by_version", {}), rubric=list(p.get("rubric", [])),
        scoring_rules=list(p.get("scoring_rules", [])), grading_policy=p.get("grading_policy", "choice_and_explanation_independent"),
        official_solution=dict(p.get("official_solution", {})), rubric_items=items,
        evidence_policy=p.get("evidence_policy", "required"),
        score_granularity=p.get("score_granularity"), rag_policy="RAG_DISABLED")


class GradeAdapter:
    #: v2: verdict target (scoring semantics). v3 (2026-08-28): shared
    #: validation moved to grade-validation-v2 (zero-side grounding, full
    #: structural id checks) and GradeResult forbids extra fields — decision
    #: (AUTO/REVIEW) semantics of scored rows changed, so runs are not
    #: comparable across the bump.
    adapter_version = "grade-bench-v3"
    #: the DEFAULT prompt version; an experiment may pin an older one to
    #: reproduce or A/B against it. Both halves of the prompt are resolved from
    #: whatever this instance carries, so a run's recorded prompt_version fully
    #: determines what was sent.
    prompt_version = "grade-v4-charitable"
    model_visible_fields = ("case_id", "pack", "selected", "transcription", "version")
    default_max_tokens = 600

    def __init__(self, role: str = "grade_primary", prompt_version: str | None = None):
        self.role = role
        self.task = "grade_escalate" if role == "grade_escalate" else "grade_primary"
        if prompt_version:
            from ..escalation import grade_system_for
            grade_system_for(prompt_version)      # refuse an unknown version early
            self.prompt_version = prompt_version

    def build_request(self, inputs: dict, bench_root: Path) -> Request:
        # Resolve BOTH halves of the prompt from this adapter's declared
        # version, so a run's recorded prompt_version fully determines what was
        # sent — and so production and the benchmark can never drift apart.
        from ..escalation import GradeResult, grade_prompt, grade_system_for
        pack = pack_from_inputs(inputs["pack"])
        blocks = grade_prompt(pack, selected=inputs.get("selected"),
                              transcription=inputs["transcription"], version=inputs.get("version"),
                              prompt_version=self.prompt_version)
        return Request(system=grade_system_for(self.prompt_version), content_blocks=blocks,
                       output_model=GradeResult,
                       prompt_version=self.prompt_version, max_tokens=self.default_max_tokens)

    def score(self, case: BenchCase, output: dict | None, error: str | None) -> dict:
        from ..escalation import GradeResult, validate_grade
        row = {"case_id": case.case_id, "split": case.split, "component": case.component,
               "schema_failure": output is None, "decision": None, "score": None,
               # False when the model-visible transcription does not cover every
               # recorded line of the answer (evidence inventory): the human label
               # judged the whole answer, so such a case is NOT an accuracy case.
               "transcription_complete": case.label.get("transcription_complete", True) is not False}
        lab = case.label
        if output is None:
            row["decision"] = "REVIEW"
            return row
        pack = pack_from_inputs(case.inputs["pack"])
        try:
            g = GradeResult(**output)
        except Exception as e:  # noqa: BLE001 — a persisted output the (now
            # stricter, extra="forbid") schema rejects is a SCHEMA FAILURE row,
            # never a scoring crash: historical outputs.jsonl rows must stay
            # scoreable without taking the whole run's scoring down
            row.update({"schema_failure": True, "decision": "REVIEW",
                        "schema_error": f"{type(e).__name__}: {str(e)[:200]}"})
            return row
        sel = case.inputs.get("selected")
        v = validate_grade(g, pack, selection_correct=lab.get("selection_correct"), selected=sel,
                           transcription=case.inputs["transcription"])
        # Evidence GROUNDING failures only. The previous substring test on
        # "evidence" also caught "evidence exceeds length limit", which is
        # verbosity, not a grounding problem — it inflated one candidate's
        # evidence-failure count by 3 in the 2026-08-25 DEV run.
        ev = v.evidence or {}
        row.update({"score": g.score, "uncertain": g.uncertain, "validation_ok": v.ok,
                    "validation_problems": list(v.problems),
                    "evidence_failure": bool(ev.get("fabricated") or ev.get("missing")
                                             or ev.get("ungrounded_credit")),
                    "evidence_ungrounded_invalid": bool(ev.get("ungrounded_invalid")),
                    "evidence_fabricated": list(ev.get("fabricated") or []),
                    "evidence_missing": list(ev.get("missing") or []),
                    "evidence_ungrounded_credit": bool(ev.get("ungrounded_credit")),
                    "evidence_verified": list(ev.get("verified") or []),
                    "decision": "AUTO" if (v.ok and not g.uncertain) else "REVIEW",
                    "met_ids": sorted(g.met_ids())})

        # ---- PRIMARY TARGET: the canonical explanation verdict ---------------
        # The raw number is a proposal production never uses as a number; it is
        # normalized through the SAME function production calls, so the
        # benchmark can never drift into its own interpretation.
        from .verdicts import verdict_from_model_score

        predicted = verdict_from_model_score(g.score, pack.max_score)
        row["predicted_verdict"] = predicted
        truth = lab.get("explanation_verdict")
        if lab.get("explanation_verdict_derivable") and truth:
            # Direction matters, not just correctness. An UPGRADE hands a student
            # credit the rubric did not earn them; a DOWNGRADE withholds credit
            # they did earn. They are different harms and are counted separately.
            rank = VERDICT_RANK
            row.update({"label_verdict": truth, "verdict_exact": predicted == truth,
                        "harmful_verdict_upgrade": rank.get(predicted, 0) > rank.get(truth, 0),
                        "harmful_verdict_downgrade": rank.get(predicted, 0) < rank.get(truth, 0)})
        else:
            row["verdict_unavailable_reason"] = (
                lab.get("explanation_verdict_reason") or "no derived verdict on this label")

        # ---- SECONDARY (derived): the final number -------------------------
        # Only meaningful once the selection state is known: the model is not
        # shown the selection and is not responsible for it.
        sc = lab.get("selection_correct")
        if sc is not None:
            from .verdicts import final_score_for

            implied = final_score_for(selection_correct=bool(sc), verdict=predicted,
                                      max_points=pack.max_score)
            row["implied_final_score"] = implied
            if lab.get("score") is not None:
                ls = float(lab["score"])
                row.update({"label_score": ls,
                            "final_exact": abs(implied - ls) < 1e-9,
                            "final_abs_error": round(abs(implied - ls), 4),
                            "harmful_upgrade": implied > ls, "harmful_downgrade": implied < ls})

        # ---- DIAGNOSTIC ONLY: raw proposal vs instructor number -------------
        # Retained for continuity with earlier runs. NOT an accuracy metric:
        # these are different quantities (see
        # docs/grade-primary-benchmark-target.md).
        if lab.get("score") is not None:
            row["raw_score_abs_delta"] = round(abs(g.score - float(lab["score"])), 4)
        if lab.get("rubric_met") is not None:
            want = set(lab["rubric_met"]); got = set(g.met_ids())
            ids = set(pack.rubric_item_ids()) or (want | got)
            row["rubric_decisions_correct"] = sum(1 for i in ids if (i in want) == (i in got))
            row["rubric_decisions_total"] = len(ids)
        if lab.get("fixed_judge_verdict") is not None:
            row["fixed_judge_verdict"] = lab["fixed_judge_verdict"]
        return row

    def aggregate(self, scored: list[dict], raw_rows: list[dict]) -> dict:
        n = len(scored)
        labeled_all = [r for r in scored if "label_score" in r]
        # accuracy only where the model saw the WHOLE answer the human graded
        labeled = [r for r in labeled_all if r.get("transcription_complete", True)]
        rub = [r for r in labeled if "rubric_decisions_total" in r]
        out = {
            "cases": n,
            "labels_available": bool(labeled),
            "labeled_excluded_transcription_incomplete": len(labeled_all) - len(labeled),
            "auto_rate_pct": _pct(sum(1 for r in scored if r["decision"] == "AUTO"), n),
            "review_rate_pct": _pct(sum(1 for r in scored if r["decision"] == "REVIEW"), n),
            "schema_failures": sum(1 for r in scored if r["schema_failure"]),
            "evidence_validation_failures": sum(1 for r in scored if r.get("evidence_failure")),
            "ungrounded_invalid_verdicts": sum(1 for r in scored
                                               if r.get("evidence_ungrounded_invalid")),
            "validation_failures": sum(1 for r in scored if r.get("validation_ok") is False),
            "usage": _usage_stats(raw_rows),
        }
        # ---- PRIMARY: explanation-verdict accuracy --------------------------
        from .verdicts import CANONICAL_VERDICTS

        judged = [r for r in scored if "label_verdict" in r
                  and r.get("transcription_complete", True)]
        out["verdict_cases"] = len(judged)
        out["verdict_cases_excluded_no_ground_truth"] = n - len(judged)
        if judged:
            out.update(_verdict_metrics(judged, CANONICAL_VERDICTS))
        else:
            out["verdict_metrics"] = (
                "unavailable: no case in this selection carries a derived explanation "
                "verdict — run the selection-correctness audit and the verdict-target "
                "dataset revision (docs/grade-primary-benchmark-target.md)")

        # ---- SECONDARY (derived): final-score agreement ---------------------
        final = [r for r in labeled if "final_exact" in r]
        if final:
            out.update({
                "final_score_cases": len(final),
                "final_score_exact_pct": _pct(sum(1 for r in final if r["final_exact"]), len(final)),
                "final_score_mae": round(statistics.mean(r["final_abs_error"] for r in final), 4),
                "harmful_upgrades": sum(1 for r in final if r["harmful_upgrade"]),
                "harmful_downgrades": sum(1 for r in final if r["harmful_downgrade"]),
            })
        else:
            out["final_score_metrics"] = ("unavailable: derived from selection correctness, "
                                          "which is not known for these cases")
        if not labeled:
            out["accuracy_metrics"] = ("unavailable: no per-item owner labels in this dataset — only "
                                       "decision/validation metrics are reported (NOT accuracy)")
        if rub:
            out["rubric_decision_correctness_pct"] = _pct(
                sum(r["rubric_decisions_correct"] for r in rub), sum(r["rubric_decisions_total"] for r in rub))
        return out


# ----------------------------------------------------------------------------
# MC / VARIANT / ALIGN (B5a/b/c)
# ----------------------------------------------------------------------------

class McResolveAdapter:
    role = "mc_resolve_cloud"
    task = "mc_resolve_cloud"
    adapter_version = "mc-bench-v1"
    prompt_version = "mc-v1"
    model_visible_fields = ("case_id", "band_png", "letters", "candidates")
    default_max_tokens = 200

    def build_request(self, inputs: dict, bench_root: Path) -> Request:
        from ..mcresolve import MC_RESOLVER_SYSTEM, MCRead, _prompt_blocks
        p = Path(bench_root) / inputs["band_png"]
        return Request(system=MC_RESOLVER_SYSTEM,
                       content_blocks=_prompt_blocks(p.read_bytes(), list(inputs["letters"]),
                                                     list(inputs["candidates"])),
                       output_model=MCRead, prompt_version=self.prompt_version,
                       max_tokens=self.default_max_tokens)

    def score(self, case: BenchCase, output: dict | None, error: str | None) -> dict:
        from ..mcresolve import MCRead, _read_ok
        row = {"case_id": case.case_id, "split": case.split, "schema_failure": output is None,
               "label": case.label.get("answer"), "label_state": case.label.get("state")}
        if output is None:
            row.update({"automatic": False, "correct": None, "unsafe_automatic": False, "abstained": True})
            return row
        read = MCRead(**output)
        auto = _read_ok(read, list(case.inputs["candidates"]))
        label = case.label.get("answer")
        row.update({"selected": read.selected, "state": read.state, "automatic": auto,
                    "correct": (read.selected == label) if label is not None else None,
                    "unsafe_automatic": bool(auto and label is not None and read.selected != label),
                    "abstained": not auto})
        return row

    def aggregate(self, scored: list[dict], raw_rows: list[dict]) -> dict:
        n = len(scored)
        auto = [r for r in scored if r["automatic"]]
        return {"cases": n, "automatic_pct": _pct(len(auto), n),
                "exact_correct_pct": _pct(sum(1 for r in auto if r["correct"]), len(auto)) if auto else None,
                "unsafe_automatic": sum(1 for r in scored if r["unsafe_automatic"]),
                "abstention_pct": _pct(sum(1 for r in scored if r["abstained"]), n),
                "schema_failures": sum(1 for r in scored if r["schema_failure"]),
                "usage": _usage_stats(raw_rows)}


class VariantResolveAdapter:
    role = "variant_resolve"
    task = "variant_resolve_cloud"
    adapter_version = "variant-bench-v1"
    prompt_version = "variant-v1"
    model_visible_fields = ("case_id", "versions", "cover_png")
    default_max_tokens = 500

    def build_request(self, inputs: dict, bench_root: Path) -> Request:
        from ..discovery import VARIANT_RESOLVE_SYSTEM, MarkerCatalog
        blocks = [{"type": "text", "text": f"Version ids: {list(inputs['versions'])}. Catalogue the variant markers."},
                  _image_block_from_file(bench_root, inputs["cover_png"])]
        return Request(system=VARIANT_RESOLVE_SYSTEM, content_blocks=blocks, output_model=MarkerCatalog,
                       prompt_version=self.prompt_version, max_tokens=self.default_max_tokens)

    def score(self, case: BenchCase, output: dict | None, error: str | None) -> dict:
        row = {"case_id": case.case_id, "split": case.split, "schema_failure": output is None}
        want_n = case.label.get("n_variants")
        if output is None:
            row.update({"automatic": False, "correct": None, "unsafe_automatic": False, "abstained": True})
            return row
        confident = bool(output.get("confident"))
        got_variants = sorted(str(m.get("variant")) for m in (output.get("markers") or []) if isinstance(m, dict))
        want_variants = sorted(str(v) for v in (case.label.get("variants") or []))
        correct = ((output.get("n_variants") == want_n) if want_n is not None else None)
        if want_variants:
            correct = bool(correct) and got_variants == want_variants if correct is not None else got_variants == want_variants
        row.update({"automatic": confident, "correct": correct,
                    "unsafe_automatic": bool(confident and correct is False), "abstained": not confident})
        return row

    aggregate = McResolveAdapter.aggregate  # same shape: exact / unsafe automatic / abstention


class AlignResolveAdapter:
    role = "align_resolve"
    task = "align_resolve_cloud"
    adapter_version = "align-bench-v1"
    prompt_version = "align-v1"
    model_visible_fields = ("case_id", "question_id", "canonical", "printed")
    default_max_tokens = 300

    def build_request(self, inputs: dict, bench_root: Path) -> Request:
        from ..alignment import ALIGN_SYSTEM, PermutationProposal
        text = ("Canonical sub-items:\n" + "\n".join(f"  {i}: {t}" for i, t in inputs["canonical"])
                + "\nPrinted in this variant:\n" + "\n".join(f"  {p}: {t}" for p, t in inputs["printed"]))
        return Request(system=ALIGN_SYSTEM, content_blocks=[{"type": "text", "text": text}],
                       output_model=PermutationProposal, prompt_version=self.prompt_version,
                       max_tokens=self.default_max_tokens)

    def score(self, case: BenchCase, output: dict | None, error: str | None) -> dict:
        row = {"case_id": case.case_id, "split": case.split, "schema_failure": output is None}
        want = {str(k): str(v) for k, v in (case.label.get("mapping") or {}).items()}
        if output is None:
            row.update({"automatic": False, "correct": None, "unsafe_automatic": False, "abstained": True})
            return row
        m = {str(k): str(v) for k, v in (output.get("printed_to_key") or {}).items()}
        printed_ids = {str(p) for p, _ in case.inputs["printed"]}
        key_ids = {str(i) for i, _ in case.inputs["canonical"]}
        complete = set(m) == printed_ids and set(m.values()) == key_ids and len(set(m.values())) == len(m)
        auto = bool(output.get("confident")) and complete
        correct = (m == want) if want else None
        row.update({"automatic": auto, "correct": correct,
                    "unsafe_automatic": bool(auto and correct is False), "abstained": not auto})
        return row

    aggregate = McResolveAdapter.aggregate


# ----------------------------------------------------------------------------

def adapter_for(role: str, prompt_version: str | None = None):
    """The adapter for a role. ``prompt_version`` pins a specific prompt (only
    the grading roles have more than one); None keeps the adapter default."""
    if role in ("grade_primary", "grade_escalate"):
        return GradeAdapter(role, prompt_version=prompt_version)
    if prompt_version:
        raise ValueError(f"role {role!r} has a single prompt version; "
                         f"cannot pin {prompt_version!r}")
    if role == "ocr_verify":
        return OcrVerifyAdapter()
    if role == "ocr_primary":
        return OcrPrimaryAdapter()
    if role == "mc_resolve_cloud":
        return McResolveAdapter()
    if role == "variant_resolve":
        return VariantResolveAdapter()
    if role == "align_resolve":
        return AlignResolveAdapter()
    raise ValueError(f"no benchmark adapter for role {role!r}")


__all__ = ["Request", "adapter_for", "OcrVerifyAdapter", "OcrPrimaryAdapter", "GradeAdapter",
           "McResolveAdapter", "VariantResolveAdapter", "AlignResolveAdapter", "BenchTranscription",
           "pack_from_inputs", "OCR_PRIMARY_PROMPTS"]
