# The deterministic risk layer (OFF / SHADOW; ACTIVE locked)

Two layers, never merged:

1. **Semantic grading** — `qwen3-vl:8b-instruct` with the frozen
   `grade-v4-charitable-local` prompt and `grade-validation-v2` validation.
   Judges explanation quality. UNCHANGED by everything below; asymmetric risk
   preferences are never implemented by making the model "stricter".
2. **Deterministic risk policy** — `autograder/riskengine.py`
   (`risk-engine-v1`) turns a validated raw verdict plus decision-time
   structural facts into a candidate action `AUTO | REVIEW | BLOCKED` under a
   frozen asymmetric loss matrix
   (`evaluation/model_selection/policies/asymmetric_grading_risk_v1.json`,
   `11e65e79e0f3…`: invalid→valid = 12 dominates; undergrades stay nonzero).

## Pipeline

```
crop/image
    -> OpenRouter OCR only            (cloud allowlist: ocr_primary/ocr_verify;
                                       UNVALIDATED — hard shipping blocker)
    -> immutable transcription
    -> local semantic grader          (grade-v4-charitable-local, Ollama)
    -> raw verdict/evidence           (never modified downstream)
    -> validation                     (grade-validation-v2)
    -> prospective deterministic
       risk policy                    (riskengine, ONLINE_OBSERVABLE inputs only)
    -> shadow AUTO/REVIEW proposal    (append-only shadow event; active grade
                                       untouched; admin-only diagnostics)
    -> human review/appeal
    -> immutable provenance           (policy/matrix/model/prompt/schema hashes
                                       on every decision)
```

SHADOW is **not** active grading: it computes and records what the policy
*would* do. Nothing is auto-finalized, no existing REVIEW is suppressed, and
the diagnostics surface is admin-only (`GET /api/admin/shadow`).

## Policy-scope taxonomy

| scope | may use | deployable online? |
|---|---|---|
| `PROSPECTIVE_DEPLOYABLE` | ONLY decision-time facts: raw verdict, schema/evidence validity, uncertainty, transcription completeness, source-integrity state, output currency, local-route availability, provenance hashes | yes (shadow now; active only after full authorization) |
| `RETROSPECTIVE_HUMAN_ASSISTED` | + reviewer disagreement, adjudicated issues | **NO — oracle-assisted replay only** |
| `ANALYSIS_BASELINE_ONLY` | decision-time facts | no (diagnostic baseline) |

**Why human disagreement is not an online feature:** reviewer disagreement
exists only *after* two humans review a case. On a new exam there are no
reviews yet, so a policy conditioned on disagreement cannot run. The engine
enforces this structurally: `ProspectiveDecisionInput.from_mapping` refuses
post-review fields by name (fail-closed on unknown fields too), retrospective
policies are `BLOCKED_NONPROSPECTIVE_POLICY` outside explicit offline
analysis, and the oracle tables in every artifact are labelled
`NOT DEPLOYABLE`. The oracle gap (e.g. AUTO risk 23 @ 67% vs prospective 34 @
85%) is the measured value of building a future *prospective* ambiguity
signal — not something to fake by leaking labels.

## Registered policies (`risk-engine-v1`)

- `prospective_valid_only_v1` — AUTO only structurally clean grounded
  `valid`; seen-46: AUTO 27/46 (58.7%), AUTO risk 20, false-full 0.
- `prospective_noninvalid_v1` — + clean `partially_valid`; AUTO 39/46
  (84.8%), risk 34, false-full 0.
- `prospective_auto_all_structurally_valid_v1` — analysis baseline.
- `retrospective_human_dispute_aware_{b,c}_v1` — oracle replays of the
  committed HUMAN_DISPUTE_AWARE policies (reproduce them exactly).

## Modes

- `off` (default): nothing happens.
- `shadow`: decision computed + appended to an idempotent, append-only
  JSONL shadow log; active grade untouched.
- `active`: code path exists, **locked**. Requires a complete
  `ActivationRecord` (exact policy/matrix/model/prompt/schema/validator
  hashes, validated OCR policy version, final-validation record, stale-check,
  and the literal owner acknowledgement) — and no production caller exists;
  tests enforce both.

## Appeal-safe provenance

Every decision carries: raw semantic verdict (immutable), candidate action,
typed reason code, policy id/version/hash, matrix name/hash, engine version,
structural state, model/prompt/schema/validator hashes, input fingerprint,
timestamp. Shadow events separate `decision_input` from
`offline_evaluation`; deleting the offline block provably changes no decision
(tested over all 138 seen-46 events).

## Rare-event sample-size limitation

Zero observed invalid→valid errors over the 5 seen invalid cases bounds the
true rate only below **45.1%** (one-sided 95%, exact). Demonstrating a rate
below 10% / 5% / 2% / 1% requires **29 / 59 / 149 / 299** independent invalid
examples with zero events (`(1-bound)^n <= 0.05`). The threshold choice is
the owner's; `autograder/rare_events.py` carries the exact math.

## OCR validation (hard blocker) and HELD_OUT

The OpenRouter OCR route has never been validated against the audited
transcriptions. The campaign is FROZEN, not executed:
`evaluation/model_selection/experiments/OCR_VALIDATION_CAMPAIGN_2026-09-02.json`
(stages, crop hashes, prompts/schemas, ≤62 calls/candidate, $2.00 hard
bound, exact future commands). HELD_OUT stays sealed until the grader,
matrix, decision policy and OCR route are all frozen and gates pass; it then
runs once via `bench final-eval`.

## Owner next steps

1. Decide the acceptable false-full bound and start accumulating invalid
   examples toward the matching sample size (59 for 5%).
2. Authorize the OCR campaign: close the two named prep items (seen46-ocr
   subset registration, per-writer WER scoring), then run stage 1 smoke.
3. Turn on SHADOW for both prospective policies in the review flow and judge
   them on shadow evidence (`RELEASE_READINESS_2026-09-02.md` tracks the
   gates; status today: SHADOW_READY).
