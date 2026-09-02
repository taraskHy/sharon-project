"""Deterministic downstream risk-decision layer — OFF / SHADOW only for now.

Position in the pipeline (the semantic grader is UNCHANGED by this layer):

    immutable transcription
        -> local semantic grader (grade-v4-charitable-local)
        -> raw GradeResult / canonical verdict
        -> validation (grade-validation-v2)
        -> deterministic risk engine (THIS MODULE)
        -> candidate action: AUTO | REVIEW | BLOCKED

The engine NEVER modifies the model's raw verdict, never re-prompts, and in
"off"/"shadow" mode never changes the active grade. "active" mode exists as a
code path but is locked: it requires a complete `ActivationRecord` (exact
policy/matrix/model/prompt/schema/validator hashes, a validated OCR policy, a
final-validation record and the literal owner acknowledgement) and has NO
production caller in this repository.

Policy-scope taxonomy (every registered policy carries exactly one):

- PROSPECTIVE_DEPLOYABLE — may consume ONLY information available at decision
  time on a brand-new exam (the ONLINE_OBSERVABLE fields below). The typed
  input `ProspectiveDecisionInput` makes misuse structurally difficult: its
  constructor fails closed on unknown fields and refuses any post-review
  target field by name.
- RETROSPECTIVE_HUMAN_ASSISTED — may additionally consume post-review human
  metadata (reviewer disagreement, adjudicated issues). Useful ONLY for
  historical replay and upper-bound analysis; the engine refuses to evaluate
  these outside explicit offline analysis, because human-disagreement
  information does not exist before review on a new case.
- ANALYSIS_BASELINE_ONLY — diagnostic baselines (structurally-gated AUTO-ALL,
  constant policies). Never deployable.

Versioned constants follow the repository idiom: historical values are never
edited in place; a semantic change bumps the version string.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional

#: risk-engine-v1 (2026-09-02): first version. OFF/SHADOW capable, ACTIVE
#: locked behind a complete ActivationRecord; five registered policies.
RISK_ENGINE_VERSION = "risk-engine-v1"

#: risk-observability-v1 (2026-09-02): first field inventory. Unknown fields
#: fail closed everywhere.
OBSERVABILITY_INVENTORY_VERSION = "risk-observability-v1"

VERDICTS = ("invalid", "partially_valid", "valid")

MODES = ("off", "shadow", "active")
DEFAULT_MODE = "off"

ACTIONS = ("AUTO", "REVIEW", "BLOCKED")

POLICY_SCOPES = ("PROSPECTIVE_DEPLOYABLE", "RETROSPECTIVE_HUMAN_ASSISTED",
                 "ANALYSIS_BASELINE_ONLY")

#: The literal, explicit owner acknowledgement ACTIVE mode requires.
ACTIVATION_ACK = "I-CONFIRM-AUTOMATIC-GRADE-APPLICATION"

# ---------------------------------------------------------------- reasons ----

REASONS = (
    "AUTO_GROUNDED_VALID",
    "AUTO_GROUNDED_PARTIAL",
    "AUTO_STRUCTURALLY_VALID_BASELINE",
    "REVIEW_INVALID_VERDICT",
    "REVIEW_PARTIAL_VERDICT",
    "REVIEW_UNCERTAIN",
    "REVIEW_SCHEMA_FAILURE",
    "REVIEW_EVIDENCE_FAILURE",
    "REVIEW_VALIDATION_FAILED",
    "REVIEW_TRANSCRIPTION_INCOMPLETE",
    "REVIEW_SOURCE_INTEGRITY",
    "REVIEW_STALE_OUTPUT",
    "REVIEW_LOCAL_GRADER_UNAVAILABLE",
    "REVIEW_WIDE_HUMAN_DISAGREEMENT",       # retrospective policies only
    "REVIEW_ACTIVE_EVIDENCE_ISSUE",         # retrospective policies only
    "BLOCKED_POLICY_UNSELECTED",
    "BLOCKED_POLICY_HASH_MISMATCH",
    "BLOCKED_MATRIX_HASH_MISMATCH",
    "BLOCKED_NONPROSPECTIVE_POLICY",
    "BLOCKED_ACTIVATION_INCOMPLETE",
)

# ---------------------------------------------------- typed refusals ---------


class RiskEngineError(RuntimeError):
    """Base class: every engine refusal is typed, never a silent grade."""


class RiskInputError(RiskEngineError):
    """A decision input violated the fail-closed field contract."""


class RiskMatrixError(RiskEngineError):
    """The loss-matrix artifact is malformed, tampered or incoherent."""


class ShadowLogError(RiskEngineError):
    """The append-only shadow log is malformed or cannot be written safely."""


# ------------------------------------------------ observability inventory ----

ONLINE_OBSERVABLE = "ONLINE_OBSERVABLE"
POST_HOC_ONLY = "POST_HOC_ONLY"
ADMIN_ONLY = "ADMIN_ONLY"
UNKNOWN = "UNKNOWN"

#: Every field any routing policy may mention, classified. Production
#: prospective decisions may consume ONLY the ONLINE_OBSERVABLE rows.
FIELD_OBSERVABILITY: dict[str, str] = {
    # available at decision time on a new exam
    "semantic_verdict": ONLINE_OBSERVABLE,
    "schema_ok": ONLINE_OBSERVABLE,
    "evidence_ok": ONLINE_OBSERVABLE,
    "validation_ok": ONLINE_OBSERVABLE,
    "uncertain": ONLINE_OBSERVABLE,
    "transcription_complete": ONLINE_OBSERVABLE,
    "source_integrity": ONLINE_OBSERVABLE,        # current | stale | issue
    "model_output_current": ONLINE_OBSERVABLE,
    "local_grader_available": ONLINE_OBSERVABLE,
    "model_digest": ONLINE_OBSERVABLE,
    "prompt_version": ONLINE_OBSERVABLE,
    "prompt_sha256": ONLINE_OBSERVABLE,
    "schema_sha256": ONLINE_OBSERVABLE,
    "validation_version": ONLINE_OBSERVABLE,
    # exists only AFTER human review / against a benchmark target
    "reviewer_verdict": POST_HOC_ONLY,
    "reviewer_disagreement": POST_HOC_ONLY,
    "wide_human_disagreement": POST_HOC_ONLY,
    "active_review_issue": POST_HOC_ONLY,
    "adjudicated_verdict": POST_HOC_ONLY,
    "final_verdict": POST_HOC_ONLY,
    "reference_verdict": POST_HOC_ONLY,
    "human_reference": POST_HOC_ONLY,
    "expected_verdict": POST_HOC_ONLY,
    "label_verdict": POST_HOC_ONLY,
    "instructor_score": POST_HOC_ONLY,
    "instructor_derived_verdict": POST_HOC_ONLY,
    "benchmark_correct": POST_HOC_ONLY,
    "model_correct": POST_HOC_ONLY,
    "strict_loss": POST_HOC_ONLY,
    # admin-side metadata, never a decision feature
    "reviewer_identity": ADMIN_ONLY,
    "reviewer_note": ADMIN_ONLY,
    "reference_source": ADMIN_ONLY,
    "aggregate_performance": ADMIN_ONLY,
}

#: substrings that mark a field as post-review target data even when the
#: exact name is not in the inventory — refusal, not silence
_POST_HOC_MARKERS = ("reference", "reviewer", "adjudicat", "instructor",
                     "disagreement", "label", "expected", "human", "benchmark",
                     "correct", "final_verdict", "loss", "target")


def classify_field(name: str) -> str:
    """Observability class of a field name; unknown fields fail closed."""
    return FIELD_OBSERVABILITY.get(name, UNKNOWN)


def observability_inventory() -> dict:
    """The versioned inventory, for artifacts and the admin surface."""
    return {"inventory_version": OBSERVABILITY_INVENTORY_VERSION,
            "fields": dict(FIELD_OBSERVABILITY),
            "rule": "production prospective decisions may consume ONLY "
                    "ONLINE_OBSERVABLE fields; unknown fields fail closed"}


# ------------------------------------------------------- decision inputs ----

_REQUIRED_BOOL_FIELDS = ("schema_ok", "evidence_ok", "validation_ok",
                         "uncertain", "transcription_complete",
                         "model_output_current", "local_grader_available")
_REQUIRED_STR_FIELDS = ("model_digest", "prompt_version", "prompt_sha256",
                        "schema_sha256", "validation_version")
_SOURCE_INTEGRITY_STATES = ("current", "stale", "issue")


@dataclass(frozen=True, slots=True)
class ProspectiveDecisionInput:
    """Exactly the ONLINE_OBSERVABLE decision-time facts — nothing else.

    Build through `from_mapping`, which fails closed: an unknown key is a
    typed refusal, and a key that names post-review target data is refused
    with an explicit POST_HOC error. This is the structural guard against
    target leakage into prospective policy evaluation.
    """
    semantic_verdict: str
    schema_ok: bool
    evidence_ok: bool
    validation_ok: bool
    uncertain: bool
    transcription_complete: bool
    source_integrity: str
    model_output_current: bool
    local_grader_available: bool
    model_digest: str
    prompt_version: str
    prompt_sha256: str
    schema_sha256: str
    validation_version: str

    def __post_init__(self):
        if self.semantic_verdict not in VERDICTS:
            raise RiskInputError(f"unknown semantic verdict "
                                 f"{self.semantic_verdict!r}; fails closed")
        if self.source_integrity not in _SOURCE_INTEGRITY_STATES:
            raise RiskInputError(f"unknown source_integrity "
                                 f"{self.source_integrity!r}; fails closed")
        for f_ in _REQUIRED_BOOL_FIELDS:
            if not isinstance(getattr(self, f_), bool):
                raise RiskInputError(f"field {f_} must be a bool, got "
                                     f"{type(getattr(self, f_)).__name__}")
        for f_ in _REQUIRED_STR_FIELDS:
            v = getattr(self, f_)
            if not isinstance(v, str) or not v:
                raise RiskInputError(f"field {f_} must be a non-empty string")

    @classmethod
    def from_mapping(cls, m: dict) -> "ProspectiveDecisionInput":
        if not isinstance(m, dict):
            raise RiskInputError("decision input must be a mapping")
        allowed = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        for key in m:
            if key in allowed:
                continue
            if classify_field(key) in (POST_HOC_ONLY, ADMIN_ONLY) or \
                    any(mk in str(key).lower() for mk in _POST_HOC_MARKERS):
                raise RiskInputError(
                    f"POST_HOC/target field {key!r} refused: post-review data "
                    "must never enter a prospective decision")
            raise RiskInputError(f"unknown field {key!r}; fails closed")
        missing = sorted(allowed - set(m))
        if missing:
            raise RiskInputError(f"missing required fields: {missing}")
        return cls(**{k: m[k] for k in allowed})

    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


@dataclass(frozen=True, slots=True)
class RetrospectiveContext:
    """Post-review metadata for RETROSPECTIVE_HUMAN_ASSISTED replay ONLY.

    This information does not exist before human review on a new case; the
    engine refuses it outside offline analysis and refuses to combine it
    with a prospective policy at all.
    """
    wide_human_disagreement: bool
    active_review_issue: bool

    def __post_init__(self):
        for f_ in ("wide_human_disagreement", "active_review_issue"):
            if not isinstance(getattr(self, f_), bool):
                raise RiskInputError(f"retrospective field {f_} must be bool")


# ------------------------------------------------------------ loss matrix ----

_REPO = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX_PATH = _REPO / "evaluation" / "model_selection" / "policies" / \
    "asymmetric_grading_risk_v1.json"


@dataclass(frozen=True, slots=True)
class MatrixRef:
    name: str
    schema_version: int
    matrix: dict
    matrix_sha256: str      # canonical hash of the matrix values alone
    policy_file_sha256: str  # the artifact's own self-hash


def _canonical_matrix_hash(matrix: dict) -> str:
    return hashlib.sha256(json.dumps(matrix, sort_keys=True).encode()).hexdigest()


def load_risk_matrix(path: Path | str = DEFAULT_MATRIX_PATH) -> MatrixRef:
    """Load and validate the frozen asymmetric loss matrix. Every failure is
    a typed `RiskMatrixError`; nothing is ever silently defaulted."""
    p = Path(path)
    try:
        raw = p.read_text(encoding="utf-8")
    except OSError as e:
        raise RiskMatrixError(f"matrix artifact unreadable: {e}") from e
    try:
        doc = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RiskMatrixError(f"matrix artifact is not valid JSON: {e}") from e
    if not isinstance(doc, dict):
        raise RiskMatrixError("matrix artifact must be a JSON object")
    for key in ("policy_name", "schema_version", "cost_matrix", "policy_sha256"):
        if key not in doc:
            raise RiskMatrixError(f"matrix artifact missing {key!r}")
    payload = json.dumps({k: v for k, v in doc.items() if k != "policy_sha256"},
                         ensure_ascii=False, sort_keys=True)
    if hashlib.sha256(payload.encode()).hexdigest() != doc["policy_sha256"]:
        raise RiskMatrixError("matrix artifact self-hash mismatch: tampered "
                              "or truncated")
    m = doc["cost_matrix"]
    if not isinstance(m, dict) or sorted(m) != sorted(VERDICTS):
        raise RiskMatrixError("cost_matrix must have exactly the three "
                              "canonical verdict rows")
    for a in VERDICTS:
        row = m[a]
        if not isinstance(row, dict) or sorted(row) != sorted(VERDICTS):
            raise RiskMatrixError(f"row {a!r} must have exactly the three "
                                  "canonical verdict columns")
        for b in VERDICTS:
            v = row[b]
            if isinstance(v, bool) or not isinstance(v, int):
                raise RiskMatrixError(f"cost[{a}][{b}] must be an integer, "
                                      f"got {type(v).__name__}")
            if v < 0:
                raise RiskMatrixError(f"cost[{a}][{b}] is negative")
        if m[a][a] != 0:
            raise RiskMatrixError(f"diagonal cost[{a}][{a}] must be zero")
    off = [m[a][b] for a in VERDICTS for b in VERDICTS if a != b]
    if not (m["invalid"]["valid"] == max(off)
            and all(m["invalid"]["valid"] > c for c in off
                    if c != m["invalid"]["valid"])):
        raise RiskMatrixError("invalid->valid must be the strictly largest cost")
    if not (m["partially_valid"]["valid"] > m["partially_valid"]["invalid"] > 0):
        raise RiskMatrixError("required ordering partially_valid->valid > "
                              "partially_valid->invalid > 0 violated")
    if not (m["valid"]["invalid"] > 0 and m["valid"]["partially_valid"] > 0):
        raise RiskMatrixError("undergrade costs must be nonzero")
    return MatrixRef(name=str(doc["policy_name"]),
                     schema_version=int(doc["schema_version"]),
                     matrix=m,
                     matrix_sha256=_canonical_matrix_hash(m),
                     policy_file_sha256=str(doc["policy_sha256"]))


# ---------------------------------------------------------------- policies ---

RuleFn = Callable[[ProspectiveDecisionInput, Optional[RetrospectiveContext]],
                  tuple[str, str]]


_SPEC_SHA_CACHE: dict[tuple[str, int], str] = {}


@dataclass(frozen=True, slots=True)
class PolicySpec:
    policy_id: str
    version: int
    scope: str
    description: str
    rule: RuleFn

    def sha256(self) -> str:
        key = (self.policy_id, self.version)
        cached = _SPEC_SHA_CACHE.get(key)
        if cached is None:
            payload = json.dumps({"policy_id": self.policy_id,
                                  "version": self.version, "scope": self.scope,
                                  "rule_source": inspect.getsource(self.rule)},
                                 sort_keys=True)
            cached = hashlib.sha256(payload.encode()).hexdigest()[:16]
            _SPEC_SHA_CACHE[key] = cached
        return cached


def _structural_gate(d: ProspectiveDecisionInput) -> Optional[tuple[str, str]]:
    """Shared deterministic structural gates, first match wins."""
    if not d.local_grader_available:
        return "REVIEW", "REVIEW_LOCAL_GRADER_UNAVAILABLE"
    if not d.model_output_current:
        return "REVIEW", "REVIEW_STALE_OUTPUT"
    if d.source_integrity != "current":
        return "REVIEW", "REVIEW_SOURCE_INTEGRITY"
    if not d.schema_ok:
        return "REVIEW", "REVIEW_SCHEMA_FAILURE"
    if not d.evidence_ok:
        return "REVIEW", "REVIEW_EVIDENCE_FAILURE"
    if not d.validation_ok:
        return "REVIEW", "REVIEW_VALIDATION_FAILED"
    if d.uncertain:
        return "REVIEW", "REVIEW_UNCERTAIN"
    if not d.transcription_complete:
        return "REVIEW", "REVIEW_TRANSCRIPTION_INCOMPLETE"
    return None


def _verdict_route(d: ProspectiveDecisionInput, *, auto_partial: bool
                   ) -> tuple[str, str]:
    if d.semantic_verdict == "valid":
        return "AUTO", "AUTO_GROUNDED_VALID"
    if d.semantic_verdict == "partially_valid":
        if auto_partial:
            return "AUTO", "AUTO_GROUNDED_PARTIAL"
        return "REVIEW", "REVIEW_PARTIAL_VERDICT"
    return "REVIEW", "REVIEW_INVALID_VERDICT"


def _rule_valid_only(d, ctx=None):
    gate = _structural_gate(d)
    if gate:
        return gate
    return _verdict_route(d, auto_partial=False)


def _rule_noninvalid(d, ctx=None):
    gate = _structural_gate(d)
    if gate:
        return gate
    return _verdict_route(d, auto_partial=True)


def _rule_auto_all_structural(d, ctx=None):
    gate = _structural_gate(d)
    if gate:
        return gate
    if d.semantic_verdict == "valid":
        return "AUTO", "AUTO_GROUNDED_VALID"
    if d.semantic_verdict == "partially_valid":
        return "AUTO", "AUTO_GROUNDED_PARTIAL"
    return "AUTO", "AUTO_STRUCTURALLY_VALID_BASELINE"


def _retro_gate(ctx: Optional[RetrospectiveContext]) -> Optional[tuple[str, str]]:
    if ctx is None:
        raise RiskInputError("retrospective policy requires a "
                             "RetrospectiveContext; refusing to guess")
    if ctx.active_review_issue:
        return "REVIEW", "REVIEW_ACTIVE_EVIDENCE_ISSUE"
    if ctx.wide_human_disagreement:
        return "REVIEW", "REVIEW_WIDE_HUMAN_DISAGREEMENT"
    return None


def _rule_dispute_aware_b(d, ctx=None):
    gate = _structural_gate(d)
    if gate:
        return gate
    retro = _retro_gate(ctx)
    if retro:
        return retro
    return _verdict_route(d, auto_partial=False)


def _rule_dispute_aware_c(d, ctx=None):
    gate = _structural_gate(d)
    if gate:
        return gate
    retro = _retro_gate(ctx)
    if retro:
        return retro
    return _verdict_route(d, auto_partial=True)


POLICY_REGISTRY: dict[str, PolicySpec] = {p.policy_id: p for p in (
    PolicySpec("prospective_valid_only_v1", 1, "PROSPECTIVE_DEPLOYABLE",
               "AUTO only a structurally clean, grounded 'valid' verdict; "
               "everything else to REVIEW", _rule_valid_only),
    PolicySpec("prospective_noninvalid_v1", 1, "PROSPECTIVE_DEPLOYABLE",
               "AUTO structurally clean 'valid' and 'partially_valid'; "
               "'invalid' always to REVIEW", _rule_noninvalid),
    PolicySpec("prospective_auto_all_structurally_valid_v1", 1,
               "ANALYSIS_BASELINE_ONLY",
               "shadow/analysis baseline: AUTO any structurally clean "
               "verdict, including 'invalid'", _rule_auto_all_structural),
    PolicySpec("retrospective_human_dispute_aware_b_v1", 1,
               "RETROSPECTIVE_HUMAN_ASSISTED",
               "valid-only + wide-human-disagreement and active-issue "
               "routing; ORACLE-ASSISTED, not deployable on new cases",
               _rule_dispute_aware_b),
    PolicySpec("retrospective_human_dispute_aware_c_v1", 1,
               "RETROSPECTIVE_HUMAN_ASSISTED",
               "valid+partial + wide-human-disagreement and active-issue "
               "routing; ORACLE-ASSISTED, not deployable on new cases",
               _rule_dispute_aware_c),
)}


def policy_table() -> list[dict]:
    """The taxonomy table (docs, tests, admin surface)."""
    return [{"policy_id": p.policy_id, "version": p.version, "scope": p.scope,
             "policy_sha256": p.sha256(),
             "online_deployable": p.scope == "PROSPECTIVE_DEPLOYABLE",
             "uses_human_or_reference_data":
                 p.scope == "RETROSPECTIVE_HUMAN_ASSISTED",
             "allowed_modes": (["off", "shadow", "active-once-unlocked"]
                               if p.scope == "PROSPECTIVE_DEPLOYABLE"
                               else ["off", "shadow (offline analysis only)"]),
             "description": p.description}
            for p in POLICY_REGISTRY.values()]


# ---------------------------------------------------------------- decisions --


@dataclass(frozen=True, slots=True)
class RiskDecision:
    action: str
    reason: str
    semantic_verdict: str
    candidate_awarded_verdict: Optional[str]
    policy_id: str
    policy_version: int
    policy_scope: str
    policy_sha256: str
    matrix_name: str
    matrix_sha256: str
    engine_version: str
    mode: str
    schema_ok: bool
    evidence_ok: bool
    uncertain: bool
    transcription_complete: bool
    source_integrity: str
    model_digest: str
    prompt_version: str
    prompt_sha256: str
    schema_sha256: str
    validation_version: str
    input_fingerprint: str
    decided_at: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RiskDecision":
        allowed = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        unknown = sorted(set(d) - allowed)
        if unknown:
            raise RiskInputError(f"unknown decision fields {unknown}; "
                                 "fails closed")
        missing = sorted(allowed - set(d))
        if missing:
            raise RiskInputError(f"missing decision fields {missing}")
        dec = cls(**d)
        if dec.action not in ACTIONS:
            raise RiskInputError(f"unknown action {dec.action!r}; fails closed")
        if dec.reason not in REASONS:
            raise RiskInputError(f"unknown reason {dec.reason!r}; fails closed")
        return dec


@dataclass(frozen=True, slots=True)
class ActivationRecord:
    """Everything ACTIVE mode requires. Any missing/mismatched field keeps
    the engine BLOCKED; no production caller constructs one."""
    owner_ack: str
    policy_id: str
    policy_sha256: str
    matrix_name: str
    matrix_sha256: str
    model_digest: str
    prompt_version: str
    prompt_sha256: str
    schema_sha256: str
    validation_version: str
    ocr_policy_version: str
    final_validation_record: str
    stale_artifacts_check_passed: bool
    configured_at: str


@dataclass(frozen=True, slots=True)
class RiskOutcome:
    """What running one case through the engine produced. `applied` is True
    only in fully-authorized ACTIVE mode; nothing in this repository sets it."""
    mode: str
    decision: Optional[RiskDecision]
    applied: bool
    active_grade_changed: bool          # ALWAYS False in off/shadow
    shadow_event_id: Optional[str]


class RiskEngine:
    def __init__(self, *, mode: str = DEFAULT_MODE, policy_id: Optional[str],
                 expected_policy_sha256: Optional[str],
                 matrix: MatrixRef,
                 expected_matrix_sha256: Optional[str],
                 activation: Optional[ActivationRecord] = None):
        if mode not in MODES:
            raise RiskInputError(f"unknown mode {mode!r}; fails closed")
        self.mode = mode
        self.policy_id = policy_id
        self.expected_policy_sha256 = expected_policy_sha256
        self.matrix = matrix
        self.expected_matrix_sha256 = expected_matrix_sha256
        self.activation = activation

    # -- pure decision ------------------------------------------------------

    def decide(self, d: ProspectiveDecisionInput,
               retrospective: Optional[RetrospectiveContext] = None, *,
               offline_analysis: bool = False,
               now: Optional[str] = None) -> RiskDecision:
        if not isinstance(d, ProspectiveDecisionInput):
            raise RiskInputError("decision input must be a "
                                 "ProspectiveDecisionInput")
        blocked = self._blocked_reason(retrospective, offline_analysis)
        spec = POLICY_REGISTRY.get(self.policy_id or "")
        if blocked:
            action, reason = "BLOCKED", blocked
        else:
            assert spec is not None
            if spec.scope == "PROSPECTIVE_DEPLOYABLE" and retrospective is not None:
                raise RiskInputError(
                    "a PROSPECTIVE_DEPLOYABLE policy must not receive "
                    "retrospective human context; refused")
            action, reason = spec.rule(d, retrospective)
            if action not in ACTIONS or reason not in REASONS:
                raise RiskEngineError(f"policy produced unknown action/reason "
                                      f"{action!r}/{reason!r}; fails closed")
        return RiskDecision(
            action=action, reason=reason,
            semantic_verdict=d.semantic_verdict,
            candidate_awarded_verdict=(d.semantic_verdict if action == "AUTO"
                                       else None),
            policy_id=self.policy_id or "UNSELECTED",
            policy_version=spec.version if spec else 0,
            policy_scope=spec.scope if spec else "UNKNOWN",
            policy_sha256=spec.sha256() if spec else "",
            matrix_name=self.matrix.name,
            matrix_sha256=self.matrix.matrix_sha256,
            engine_version=RISK_ENGINE_VERSION,
            mode=self.mode,
            schema_ok=d.schema_ok, evidence_ok=d.evidence_ok,
            uncertain=d.uncertain,
            transcription_complete=d.transcription_complete,
            source_integrity=d.source_integrity,
            model_digest=d.model_digest, prompt_version=d.prompt_version,
            prompt_sha256=d.prompt_sha256, schema_sha256=d.schema_sha256,
            validation_version=d.validation_version,
            input_fingerprint=d.fingerprint(),
            decided_at=now if now is not None
            else time.strftime("%Y-%m-%d %H:%M:%S"))

    def _blocked_reason(self, retrospective, offline_analysis) -> Optional[str]:
        spec = POLICY_REGISTRY.get(self.policy_id or "")
        if spec is None:
            return "BLOCKED_POLICY_UNSELECTED"
        if self.expected_policy_sha256 != spec.sha256():
            return "BLOCKED_POLICY_HASH_MISMATCH"
        if self.expected_matrix_sha256 != self.matrix.matrix_sha256:
            return "BLOCKED_MATRIX_HASH_MISMATCH"
        # retrospective policies need post-review data that does not exist in
        # production — they run ONLY through explicit offline analysis
        if spec.scope == "RETROSPECTIVE_HUMAN_ASSISTED" and not offline_analysis:
            return "BLOCKED_NONPROSPECTIVE_POLICY"
        # ACTIVE mode accepts PROSPECTIVE_DEPLOYABLE policies exclusively
        if spec.scope != "PROSPECTIVE_DEPLOYABLE" and self.mode == "active":
            return "BLOCKED_NONPROSPECTIVE_POLICY"
        if self.mode == "active" and self._activation_missing(spec):
            return "BLOCKED_ACTIVATION_INCOMPLETE"
        return None

    def _activation_missing(self, spec: PolicySpec) -> list[str]:
        a = self.activation
        if a is None:
            return ["activation_record"]
        missing = []
        if a.owner_ack != ACTIVATION_ACK:
            missing.append("owner_ack")
        if a.policy_id != spec.policy_id or a.policy_sha256 != spec.sha256():
            missing.append("policy_hash")
        if a.matrix_name != self.matrix.name or \
                a.matrix_sha256 != self.matrix.matrix_sha256:
            missing.append("matrix_hash")
        for f_ in ("model_digest", "prompt_version", "prompt_sha256",
                   "schema_sha256", "validation_version", "ocr_policy_version",
                   "final_validation_record", "configured_at"):
            if not getattr(a, f_):
                missing.append(f_)
        if a.stale_artifacts_check_passed is not True:
            missing.append("stale_artifacts_check")
        return missing

    # -- mode-aware application --------------------------------------------

    def run_case(self, case_id: str, model_run_id: str,
                 d: ProspectiveDecisionInput,
                 retrospective: Optional[RetrospectiveContext] = None, *,
                 offline_analysis: bool = False,
                 shadow_log: Optional["ShadowLog"] = None,
                 offline_evaluation: Optional[dict] = None,
                 now: Optional[str] = None) -> RiskOutcome:
        """OFF: nothing happens. SHADOW: decide + append an event; the active
        grade is NEVER touched. ACTIVE: fully-authorized decisions only; this
        repository contains no production caller."""
        if self.mode == "off":
            return RiskOutcome(mode="off", decision=None, applied=False,
                               active_grade_changed=False, shadow_event_id=None)
        decision = self.decide(d, retrospective,
                               offline_analysis=offline_analysis, now=now)
        event_id = None
        if self.mode == "shadow":
            if shadow_log is not None:
                event = build_shadow_event(case_id, model_run_id, d, decision,
                                           offline_evaluation)
                shadow_log.append(event)
                event_id = event["event_id"]
            return RiskOutcome(mode="shadow", decision=decision, applied=False,
                               active_grade_changed=False,
                               shadow_event_id=event_id)
        applied = decision.action == "AUTO"
        return RiskOutcome(mode="active", decision=decision, applied=applied,
                           active_grade_changed=applied,
                           shadow_event_id=None)


# ------------------------------------------------------------ shadow log -----

#: Field-separation contract for shadow events: the runtime policy consumes
#: ONLY `decision_input`; `offline_evaluation` exists for offline scoring and
#: deleting it must never change the decision.
SHADOW_EVENT_VERSION = "shadow-event-v1"


def shadow_event_id(case_id: str, policy_sha256: str,
                    input_fingerprint: str) -> str:
    payload = json.dumps({"case_id": case_id, "policy_sha256": policy_sha256,
                          "input_fingerprint": input_fingerprint},
                         sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def build_shadow_event(case_id: str, model_run_id: str,
                       d: ProspectiveDecisionInput, decision: RiskDecision,
                       offline_evaluation: Optional[dict] = None) -> dict:
    return {"event_version": SHADOW_EVENT_VERSION,
            "event_id": shadow_event_id(case_id, decision.policy_sha256,
                                        decision.input_fingerprint),
            "case_id": case_id,
            "model_run_id": model_run_id,
            "decision_input": asdict(d),
            "decision": decision.to_dict(),
            "offline_evaluation": offline_evaluation}


def replay_decision_from_event(event: dict, engine: RiskEngine) -> RiskDecision:
    """Recompute the decision from DECISION INPUT FIELDS ONLY — proves the
    offline evaluation block is not load-bearing."""
    d = ProspectiveDecisionInput.from_mapping(event["decision_input"])
    retro = None
    scope = POLICY_REGISTRY.get(engine.policy_id or "")
    offline = False
    if scope is not None and scope.scope == "RETROSPECTIVE_HUMAN_ASSISTED":
        raise RiskInputError("shadow events for retrospective policies are "
                             "replayed only through the offline analysis path")
    return engine.decide(d, retro, offline_analysis=offline,
                         now=event["decision"]["decided_at"])


class ShadowLog:
    """Append-only JSONL shadow-event log with idempotent event ids.

    Duplicate ids are skipped (typed result False), malformed lines are a
    typed refusal naming the line, and writes are single-line appends under
    a lock so concurrent writers cannot interleave."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._ids: set[str] = set()
        if self.path.exists():
            for i, line in enumerate(
                    self.path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    ev = json.loads(line)
                    eid = ev["event_id"]
                except (json.JSONDecodeError, KeyError, TypeError) as e:
                    raise ShadowLogError(
                        f"malformed shadow event at line {i} of {self.path}: "
                        f"{e}") from e
                self._ids.add(eid)

    def append(self, event: dict) -> bool:
        if not isinstance(event, dict) or "event_id" not in event or \
                event.get("event_version") != SHADOW_EVENT_VERSION:
            raise ShadowLogError("refusing to append a malformed shadow event")
        line = json.dumps(event, ensure_ascii=False, sort_keys=True)
        with self._lock:
            if event["event_id"] in self._ids:
                return False
            try:
                with self.path.open("a", encoding="utf-8", newline="\n") as f:
                    f.write(line + "\n")
            except OSError as e:
                raise ShadowLogError(f"shadow log not writable: {e}") from e
            self._ids.add(event["event_id"])
            return True

    def events(self) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        for i, line in enumerate(
                self.path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ShadowLogError(
                    f"malformed shadow event at line {i}: {e}") from e
        return out


def build_engine(*, mode: str = DEFAULT_MODE, policy_id: str,
                 matrix_path: Path | str = DEFAULT_MATRIX_PATH,
                 activation: Optional[ActivationRecord] = None) -> RiskEngine:
    """Convenience constructor pinning the expected hashes to the CURRENT
    registry/matrix — callers that must pin historical hashes construct
    RiskEngine directly."""
    matrix = load_risk_matrix(matrix_path)
    spec = POLICY_REGISTRY.get(policy_id)
    return RiskEngine(mode=mode, policy_id=policy_id,
                      expected_policy_sha256=spec.sha256() if spec else None,
                      matrix=matrix,
                      expected_matrix_sha256=matrix.matrix_sha256,
                      activation=activation)
