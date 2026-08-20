# Production reliability + low-review layer

Implemented 2026-08-20 on the laptop. **No model was run**: no Qwen, no
Ollama, no Gemini, no OpenRouter, no RAG index build, no OCR, no benchmark.
Everything below is deterministic code, mocks and small synthetic fixtures.

The layer sits *around* the validated pipeline. Its purpose is to raise
grading reliability and cut human REVIEW **without** lowering the bar for
what may be auto-accepted: a case is only removed from REVIEW when
deterministic validation, a second independent read, or an exactly identical
prior decision settles it.

---

## 1. Evidence-grounded grading — `evidence.py`

A rubric item may not be credited on an unsupported semantic assertion. The
grader returns, per item, whether it is met and a SHORT span copied from the
frozen student transcription; the span is then verified deterministically.

* Normalisation covers only harmless protocol differences — unicode form,
  bidi controls, Hebrew niqqud, whitespace runs, quote/dash variants, Latin
  case, and the punctuation a grader wraps a quote in. It never reorders or
  rewrites letters, so a fabricated sentence cannot normalise into a real one.
* The transcription is **read-only**. Verification builds a temporary copy;
  nothing is ever written back (asserted byte-for-byte by test).
* Fabricated or missing evidence ⇒ the grading result is INVALID ⇒ grading
  escalation. It is never accepted, and the transcription is never edited to
  make it match.
* Items that legitimately need no quoted span declare
  `RubricItemSpec(requires_evidence=False)`. The exemption is per item, in
  the rubric schema — the check is never weakened globally.

## 2. OCR confidence vs grading confidence — `signals.py`

Two different failure classes, one routing rule:

| OCR | grading | route |
|---|---|---|
| `OCR_UNRESOLVED` | anything | OCR resolution/escalation, else REVIEW |
| `OCR_OK` | `GRADE_OK` / none | AUTO |
| `OCR_OK` | `GRADE_UNCERTAIN` / `GRADE_INVALID` | grading escalation (optionally with RAG), else REVIEW |
| `OCR_OK` | `GRADE_DISAGREEMENT` | REVIEW (more grading calls cannot settle it) |
| any | any | `PAUSED` when the budget stopped model work |

A stronger OCR pass is never bought because grading is hard, and a stronger
grader is never bought because the crop is unreadable.

## 3. Empirical confidence infrastructure — `signals.DecisionSignals`

Raw MC / OCR / grading signals are captured and persisted (CV score and
margin, candidate cells, agreement flags, crop-quality verdict, verifier
result, output length, script anomalies, schema validity, evidence counts,
invariant results, primary/escalation agreement, declared uncertainty,
model-reported confidence).

**No threshold is fitted here.** Model-reported confidence is one recorded
signal, never sufficient evidence for AUTO. The point is to make a future
calibration possible without an architectural change.

## 4. Pre-OCR image triage — `imagequality.py`

Deterministic pixel statistics and geometry, no model: `OK`, `BLANK`,
`LOW_CONTRAST`, `CLIPPED`, `EXTREME_SKEW`, `SUSPICIOUS_CROP`, `INVALID`.
Contrast is measured as ink-vs-paper separation (median paper level minus the
darkest 0.5 %), *not* histogram spread — the latter rejects perfectly
readable sparse handwriting. `triage_with_recovery()` runs the cheap
deterministic recoveries first (widen the crop, re-render), injected by the
caller, and only then escalates. `should_call_ocr()` stops the pipeline from
spending a call to discover that a crop is missing or blank.

## 5. Batch-level anomaly detection — `anomaly.py`

Per-item confidence structurally cannot see a wrong template, a shifted crop,
an inverted variant mapping or a wrong key column. These are visible only
across students. Codes: `QUESTION_BLANK_RATE_SPIKE`, `CROP_FAILURE_CLUSTER`,
`MC_AMBIGUITY_SPIKE`, `OCR_FAILURE_CLUSTER`, `OCR_LENGTH_ANOMALY`,
`QUESTION_SCORE_DEGENERATE`, `GRADE_REVIEW_CLUSTER`, `GRADE_INVALID_CLUSTER`,
`ALIGNMENT_FAILURE_CLUSTER`, `VARIANT_DISTRIBUTION_ANOMALY`,
`VARIANT_UNRESOLVED_RATE`, `PAGE_COUNT_MISMATCH_CLUSTER`,
`TEMPLATE_MISMATCH_CLUSTER`.

One systemic failure ⇒ **one** warning naming the affected
question/page/template/variant and the number of students — not one review
per student. Nothing fires below `min_exams`/`min_affected`, and an outlier
must be both absolutely high and clearly apart from the rest of the batch.
Detection never changes a grade.

## 6. Deterministic grade invariants — `invariants.py`

Range, known rubric ids, no double credit, component sums, caps, declared
mutual exclusion, declared prerequisites, `wrong_choice_zero`, `choice_only`,
score granularity, and `total = sum of question scores`.

A violation marks the result INVALID and escalates. The **only** sanctioned
repair is purely arithmetic and explicitly opt-in: `repair_arithmetic()` (when
every credited item declares its points, the score is a sum with no judgement
in it) and `recompute_exam_totals()` (the exam total *is* the sum of decided
question scores). Neither runs inside validation.

`grade_exam()` runs the exam-level check on every result: the arithmetic
there is plain Python, so a violation is a real defect and is routed to a
human rather than corrected.

## 7. Pre-run cost / query estimator — `estimate.py`

Exam structure + policies + configured routes ⇒ expected calls, tokens and
cost, per job and per exam. No provider call; pricing comes from a local
`[pricing]` table in `models.toml` and is never fetched from the network.
Partial pricing refuses to guess a total. Escalation rates are **assumptions**
with conservative defaults, overridable from an earlier run's measured
metrics; the source is reported so an assumption is never read as a
measurement. Everything is labelled `ESTIMATE`; actual usage still comes only
from the ledger.

## 8. Drift protection + canary — `provenance.py`, `canary.py`

Provenance per model-backed decision: task, requested model, backend, prompt
version + hash, schema hash, decoding, reasoning, pack hash, per-input hashes,
reported provider/model, request and generation id, cache hit, timestamp —
plus a configuration fingerprint and `drift_between()`. Secrets are filtered
structurally.

The canary is a **mechanism and config contract only**: frozen suite →
candidate run through an injected runner → comparison → promotion rule, with
separate suites for `mc_resolver`, `ocr` and `grading`. No suite is populated
in this repository, and nothing here executes a model. An empty suite can
never authorise a promotion.

## 9. Privacy minimisation — `privacy.py`

Provider payloads are built by whitelist, so an identifying field added
upstream cannot leak downstream. OCR sends the crop plus an anonymous item
id; grading sends the item id, question/rubric context, the frozen student
text, the selected option and the cited evidence. `ModelGateway` refuses any
request whose content blocks carry an identifying key and records path-like
strings as warnings; ledger entries are scrubbed and the gateway records only
whitelisted metadata, so a filename or filesystem path cannot reach
`usage.jsonl`.

## 10. Exact reuse, no semantic learning — `reuse.py`

Reuse is allowed on exact mechanical identity only: variant marker, package
template, question pack, byte-exact image, alignment, page structure. One
differing fact is a miss. Semantic reuse is refused at every entry point
(`SemanticReuseRefused`), including a named `reuse_grade_by_similarity()` stub
that exists so the refusal is explicit and greppable. Two students writing the
same sentence never share a score.

## 11–13. Review priority, reason codes, grouping — `reviewqueue.py`

* Every REVIEW item carries a stable code (`MC_UNRESOLVED`, `MC_CONFLICT`,
  `OCR_UNRESOLVED`, `OCR_PROVIDER_DISAGREEMENT`, `GRADE_INVALID`,
  `GRADE_UNCERTAIN`, `GRADE_DISAGREEMENT`, `EVIDENCE_INVALID`,
  `VARIANT_UNRESOLVED`, `ALIGNMENT_UNRESOLVED`, `PACKAGE_ANOMALY`,
  `BUDGET_PAUSED`, `PROVIDER_FAILED`) and a concise explanation rendered from
  recorded facts — never generated prose.
* Priority: systemic/package → high-point grade uncertainty → unresolved
  selection under `wrong_choice_zero` → reading disagreement → low-impact
  ambiguity. **Order only**; no grade is touched, and the order is stable
  under input shuffling.
* Grouping is by exact mechanical fingerprint. `apply_to_all` takes the
  persisted scope, applies to exactly the matching exams, and logs the
  decision with its fingerprint in `apply_to_all.jsonl`. Semantic grading
  decisions can never be broadcast.

## 14–15. Decision records + early-exit accounting — `trace.py`

One compact record per question route: final state, reason, stages executed,
stages skipped **and why**, deterministic decisions, signals, grading policy,
variant/alignment source, pack hash, and every model call with its request id,
tokens and cost. `EarlyExitLedger` aggregates explanations skipped,
OCR/grading/MC/cloud calls avoided, cache hits, REVIEW cases avoided by
automatic escalation, and the share of questions graded fully locally, broken
down by skip reason.

## 16–17. Grading RAG policies + pack auditability — `gradingpack.py`

`RAG_DISABLED` / `RAG_ALWAYS` / `RAG_ON_UNCERTAIN` / `RAG_ON_ESCALATION`.
`RAG_ALWAYS` (today's behaviour) remains the default **because which policy is
most efficient has not been measured** — see the strong-PC plan below.
Retrieval stays grading-side, local, top-k-bounded, character-budgeted and
provenance-tagged, and the query is built from question + rubric only, so it is
never steered by the student's words. It is never OCR repair.

Each pack records question/rubric/solution hashes, grading and RAG policy,
chunk ids, retrieval scores, index fingerprint, character/token budget and its
own version + hash. Changing the rubric, the solution, the course source or
the retrieval configuration invalidates the pack and every cache key derived
from it.

## 18. Package preflight — `preflight.py`

Run once after discovery, before any student exam: variant ids valid, unique
and known to the key; every variant's alignment complete and bijective; no
duplicate canonical assignment; key/rubric question ids consistent; no
duplicate ids; maxima present and coherent; an accepted, deterministically
verified answer for every variant; valid grading policies; required template
regions present; deterministic total score.

A structural defect returns `PACKAGE_SETUP_REQUIRED` with the exact unresolved
facts and what is needed for each — instead of grading 180 exams and producing
180 identical reviews.

---

## Integration status

| Component | Status |
|---|---|
| Package preflight | **A** — in `orchestrator.prepare_exam_package` and the UI |
| Privacy filtering | **A** — enforced inside `ModelGateway.call` and the ledger |
| Review reason codes / priority / grouping | **A** — `reviewui` + web UI |
| Batch anomaly detection | **A** — batch view over persisted results |
| Grade invariants (exam level) | **A** — `grade_exam` self-check |
| Image triage (undecodable crop) | **A** — `mcresolve.resolve_row` |
| Cost estimator | **A** — job view (needs a parsed `answer_key.json`) |
| Evidence-grounded grading | **B** — enforced in `escalation.validate_grade`, which the live CLI path does not yet call |
| Grade invariants (question level) | **B** — same path as above |
| OCR/grading status separation + routing | **B** — mock-tested; the live path still uses the legacy batch judge |
| DecisionSignals capture | **B** — produced by the escalation engine, consumed by nothing live yet |
| Image triage (blank/contrast/clip/skew) | **B** — not yet called before the OCR/explanation pass |
| Decision traces + early-exit accounting | **B** — recorded on demand; the pipeline does not yet write `decisions.jsonl`, and the UI says so when it falls back |
| RAG grading policies | **B** — routed inside `escalate_grade`; policy choice unmeasured |
| Exact reuse store | **B** — mechanism + tests; not yet consulted by the discovery path |
| Canary suites | **C/D** — mechanism only; no suite populated, deliberately |
| Model/prompt provenance | **C** — recorded correctly from a `CallResult`; real provider fields (`provider`, `generation_id`) are unverified against a live response |

**A** = in the production route · **B** = implemented and integration-tested
with mocks · **C** = infrastructure exists, real model behaviour unvalidated ·
**D** = deferred to strong-PC empirical work.

The honest headline: the **escalation engine (and therefore evidence-grounded
grading, typed statuses and signal capture) is still not wired into
`cli.run_grade_pipeline`**, which continues to judge explanations in batch via
`ExplanationJudgement`. That rewire changes grading behaviour and cannot be
validated without a model, so it was not attempted here.

---

## Tests to run on the strong PC (not run here)

None of these may run on the laptop: each needs a GPU, a real provider, or a
large batch.

### T1 — RAG grading-policy A/B (§16)
*Objective:* choose between `RAG_ON_UNCERTAIN`, `RAG_ON_ESCALATION` and
`RAG_ALWAYS` on decision preservation vs cost, instead of defaulting.
*Command:* `python scripts/grading_rag_ab.py` (paired harness, already frozen)
*Needs:* OpenRouter (or a local 27B on GPU), the bge-m3 index, ~2 provider
calls per item per arm. *Resources:* GPU ≥ 12 GB if local; otherwise network
only.

### T2 — Evidence-grounded grading fidelity (§1)
*Objective:* how often does a real grader cite a span that does not exist, and
how much does the check raise REVIEW? Measure fabrication rate and the
false-invalid rate on the audited batch.
*Needs:* grading model, ~1 call per written answer (~130 items).
*Blocking question:* if fabrication is common, the escalation budget must be
sized for it before the engine is wired into the live path.

### T3 — Image-triage threshold calibration (§4)
*Objective:* on real crops, confirm that no legitimate handwriting is
classified `BLANK`/`LOW_CONTRAST`, and measure how many OCR calls the triage
actually saves. *Needs:* the 129-item OCR benchmark crops, CPU only, no model.
This is the cheapest of the five and should run first.

### T4 — Confidence calibration from persisted signals (§3)
*Objective:* fit AUTO/REVIEW thresholds on `DecisionSignals` against
owner-verified outcomes. *Needs:* a graded batch WITH signals persisted, i.e.
T2 must run first. CPU only.

### T5 — Canary population + drift baseline (§8)
*Objective:* freeze a small canary per task (MC / OCR / grading) and record its
accepted baseline + provenance, so a later model or routing change can be
accepted or rejected on evidence. *Needs:* one run per suite against the
production models (~20–40 provider calls total). Must be done once **before**
any model change, not after.

### T6 — Batch-anomaly threshold check (§5)
*Objective:* replay `detect_batch_anomalies` over the historical prob batch and
confirm it stays silent on a known-good batch and fires on the known A2/A3
misalignment. *Needs:* CPU only, persisted results — no model. Run with T3.
