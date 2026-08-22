# OCR verifier source audit — 2026-08-21

**Audited:** the handwritten-text/OCR verifier — `autograder/escalation.py` `escalate_ocr`
(`OCR_VERIFY_SYSTEM`, `OCRVerifyResult`, `ocr_suspicion`, `OCRDecision`) and its sole
production call site in `autograder/reliability.py` `_decide_item` / `run_reliability_judging`.

**Method:** 6 read-only cluster auditors (V1 inputs/crops, V2 prompt/schema, V3 content
semantics, V4 infra, V5 failure modes, V6 downstream) over 25 contract points (45
sub-findings), followed by adversarial re-verification of all 17 defect/BROKEN claims.
Source-only; no model calls. Adversarial verdicts override the original classification
wherever they differ; the table below shows the final adjudicated classifications.

**Intended contract:** the verifier receives exactly two inputs — the crop image of the
student's handwriting and the frozen transcription — and judges transcription fidelity
only (omissions, additions, substitutions). No reference solution, rubric, answer key,
RAG content, or CER/WER data may reach it; it must never solve, correct, or improve the
text (`escalation.py:82-91`).

**Status of fixes:** F1–F8 below were applied after the audit (commit pending). The table
keeps the original audit-record classification; the "fixed?" column reflects the applied fix.

## Headline findings

1. **The verifier is UNWIRED in production.** No code anywhere builds the
   `(question_id, sub_item_id) -> crop` dict; the only production call to
   `run_reliability_judging` (`cli.py:942-948`) passes no `crops=`, so `crop` is always
   `None` and `escalate_ocr` returns REVIEW "suspicious; no verifier available"
   (`escalation.py:131-134`) before the model call at `escalation.py:142-145` can ever run.
   Every suspicious read becomes advisory human REVIEW (score unaffected); the
   quality-triage gate, the verify→AUTO recovery, and the ocr_verify budget line
   (`estimate.py:80,162`) are dead code. Fail-safe (never fail-open), but the escalation
   route the system is designed and costed around cannot occur. Not fixed — next phase.
2. **The input surface is contract-clean.** `escalate_ocr`'s signature has no parameter
   that could carry a pack/key/rubric (`escalation.py:115-117`); the request is exactly
   crop + "Proposed transcription:" text; meta feeds only cache/ledger/budget, never the
   provider (`gateway.py:231-232`); the transcription is passed through unmutated on every
   branch and no later stage can write a "corrected" reading back (points 1a, 2a, 16, 25a-b).
3. **Live defects found and fixed (F1–F8):** swallowed BudgetExceeded (F1), AUTO-eligible
   omitted confidence (F2), discarded partial/illegible self-report (F3), digit/operator
   suspicion blind spot (F4), empty-transcription fail-open before crop triage (F5),
   ledger attribution meta (F6), verifier trace metadata + failed-call tracing (F7),
   task-name forwarding (F8).
4. **What stays empirical (UNTESTED_EMPIRICALLY):** prompt effectiveness, confidence
   calibration, AUTO-gate strictness, and suspicion-signal calibration are real-model
   properties with no empirical validation in this repo. The prepared signal-2 benchmark
   (`scripts/m2_verify_run.py`) must run before any AUTO decision from the verifier is trusted.

## Fixes applied (post-audit, commit pending)

- **F1** `escalate_ocr` re-raises `BudgetExceeded`; `reliability.py` wraps the call →
  PAUSED/BUDGET_PAUSED (mirrors `escalate_grade`, `escalation.py:326-329,397-398`).
- **F2** `OCRVerifyResult.confidence` default `"medium"` → `"low"`: an omitted
  self-assessment can no longer pass the AUTO gate (`escalation.py:79,153`).
- **F3** legibility `partial`/`illegible` with non-empty text now feeds `extra_suspicion`
  (`self_declared_partial`/`self_declared_illegible`) → advisory REVIEW flag, score
  unaffected; `primary_legibility` persisted on `OCRSignals`.
- **F4** `ocr_suspicion` `short_technical_token` also fires on digits/operators (dead
  `_DIGIT_TOKEN` at `escalation.py:40` wired in).
- **F5** empty transcription: crop + quality consulted first — non-blank crop → REVIEW
  OCR_UNRESOLVED; legibility `full` with empty text → REVIEW; blank/no crop → AUTO
  "missing" as before (reorders `reliability.py:369-377`).
- **F6** gateway meta for `escalate_ocr` AND `escalate_grade` now carries
  `job_id`/`exam_id` (ledger attribution; `reliability.py:381,433`).
- **F7** `OCRDecision` gained `attempted`/`error`/`call_meta`; executed ocr_verify traces
  carry model/cache_hit/usage/request_id/latency/cloud (via `is_cloud_route`);
  attempted-but-failed calls are traced FAILED with no avoided-credit.
- **F8** `_decide_item` forwards `config.ocr_verify_task` to `escalate_ocr`
  (trace/task-name consistency).

## Full 25-point audit table

Classifications are the final adjudicated ones (post-adversarial). Sub-entries that
differ are listed with their letter.

| # | point | classification | defect? | fixed? | finding |
|---|-------|----------------|---------|--------|---------|
| 1 | verifier input surface; production call-site values | a: HOLDS_WITH_CAVEAT; b: UNWIRED | no | F8 | Input surface contract-clean: crop + frozen transcription + anonymous meta only; no pack/key parameter exists (`escalation.py:115-117,142-145`). Sole production caller is `cli.py:942` (not orchestrator.py) and passes no crops. Task-name mismatch (hardcoded `'ocr_verify'` vs `config.ocr_verify_task`) fixed by F8. |
| 2 | fidelity-only prompt; "ONE line/cell" premise | a: HOLDS_WITH_CAVEAT; b: UNWIRED | yes (2b) | — | `OCR_VERIFY_SYSTEM` is fidelity-only, no course content leaks (`escalation.py:82-91`); the "ONE handwritten line/cell" premise has no call-site enforcement and the verifier call is production-unreachable (crops root cause). |
| 3 | `OCRVerifyResult` schema + parse path | HOLDS_WITH_CAVEAT | no | F2 (caveat b) | Fail-closed end to end: local pydantic validation, bounded repair, hard `BackendError`, exceptions → REVIEW (`escalation.py:146-149`); corrupt cache = miss. Caveat that minimal `{"verdict":"supported"}` cleared the AUTO gate is closed by F2. |
| 4 | verdict semantics: AUTO vs REVIEW gate | HOLDS_WITH_CAVEAT | no | F7 (caveat 4) | AUTO iff supported + confidence high/medium + all three lists empty (`escalation.py:153-156`); everything else REVIEW, incl. self-inconsistent output. Unused looser `signals.ocr_status_from` (`signals.py:185-194`) is a latent inconsistency; trace-metadata caveat fixed by F7. |
| 5 | confidence semantics; omitted default | a: UNTESTED_EMPIRICALLY; b: HOLDS_WITH_CAVEAT | yes (5b) | F2 | Confidence is a raw model self-report; calibration deliberately absent (`signals.py:14-17,100-106`) — empirical. Omitted confidence defaulted to `"medium"` and passed the AUTO gate (`escalation.py:79,153`) — unsafe default on a safety gate, fixed by F2. |
| 6 | omission handling | UNWIRED | yes | — | Mechanism fail-closed as written (any omission → REVIEW) but verifier unreachable (crops root cause, shared with 7/8/10/12b). Omission-laden but plausible Hebrew ≥3 chars trips no text signal → AUTO, image never consulted. |
| 7 | substitution handling | UNWIRED | no | — | Any reported substitution would REVIEW; `short_technical_token`/`no_hebrew` cover the flagged classes. Blind spot: within-Hebrew word swaps and mixed text ≥40 chars trip no deterministic signal. Same crops root cause (not double-counted). |
| 8 | addition/fabrication handling | UNWIRED | no | — | `repetition`/`protocol_artifact` run in production; the strongest guard — BLANK/INVALID crop blocks grading (`escalation.py:122-127`, `reliability.py:169,403-414`) — unreachable. Fabricated plausible Hebrew over an empty region can AUTO. |
| 9 | uncertain handwriting (legibility) | a: HOLDS; b: BROKEN | yes (9b) | F3 | Empty text + illegible/partial → REVIEW, grader skipped (`reliability.py:354-368`) — correct. But partial/illegible WITH text bypassed the guard and the legibility variable was never read again (dead after `reliability.py:356`) — the model's own structured uncertainty discarded; fixed by F3 (advisory flag + persisted `primary_legibility`). |
| 10 | Hebrew/English mixed text | HOLDS_WITH_CAVEAT | no | — | `no_hebrew` (>12 chars) and `short_technical_token` (<40 chars) cover short mixed text; flagging can only add review, never remove points (`reliability.py:415-418,478-481`). Every long pure-English answer is guaranteed human review (verifier that could clear it can't run); `hebrew_expected` knob is dead configurability. |
| 11 | formulas/symbols (digits/operators) | UNWIRED | yes | F4 | `_DIGIT_TOKEN` compiled but never consulted (`escalation.py:40`, sole occurrence); short formulas (`x=3`, `5+3=8`) tripped zero signals → AUTO, image never consulted — "the known dangerous class" per the code's own comment. Independent of crops wiring. Fixed by F4. |
| 12 | empty transcription vs crop; blank crop vs text | a: BROKEN; b: UNWIRED | yes (12a) | F5 | 12a: empty text returned AUTO "missing" with OCR_OK stamped BEFORE crop lookup (`reliability.py:369-377`) — fail-open, deterministic triage bypassed; fixed by F5. 12b: BLANK/INVALID-crop guard is correct and fail-closed in source but unreachable (crops). |
| 13 | crops producer; keying/staleness | a: UNWIRED; b: HOLDS_WITH_CAVEAT | yes (13a) | — | **Central finding:** nobody builds the crops dict — not cli.py, not webui.py's review screen (`webui.py:904`); only tablecrop's MC bands exist (different path). Keying convention well-defined (canonical ids post-remap); missing/bad crop fail-safe; no provenance/content binding on a future builder. |
| 14 | preprocessing; red-ink masking | a: HOLDS_WITH_CAVEAT; b: UNWIRED | no | — | Crop transmitted byte-identical to the provider (`escalation.py:142-143`); `triage_crop` is read-only; `triage_with_recovery` has no production caller. No masking guarantee attaches to future verifier crops: `--mask` is opt-in on the plain CLI (`cli.py:1200-1201`), jobs default it on (`jobs.py:159,298-299`); a builder rendering from source files would leak instructor red ink. |
| 15 | gateway routing; reachability | a: HOLDS; b: UNWIRED | no | — | Unconfigured/disabled `ocr_verify` route fails safe: `ocr_available=False`, gateway=None, REVIEW never AUTO (`reliability.py:218,260-265`, `gateway.py:180-189`). The model call itself is production-unreachable (crops). |
| 16 | privacy: payload, meta, ledger | HOLDS_WITH_CAVEAT | no | F6 (partial) | Payload minimal and anonymous (sha256 `item_id`); meta never reaches the provider; privacy scan pre-send (`gateway.py:212-216,231-232`). Whitelist builders (`privacy.build_ocr_request` etc.) are test-only — the guarantee rests on call-site discipline. Ledger attribution gap fixed by F6; ledger whitelist still omits `item_id`. |
| 17 | request-cache fingerprint | HOLDS_WITH_CAVEAT | no | — | Fingerprint covers crop b64, transcription text, system prompt, output schema, route fields (`requestcache.py:28-46`); failures never cached; corrupt entry = miss. Caveat: `base_url` and openrouter `provider` absent from `fingerprint_fields` (`gateway.py:96-104`) — same-name routes on different endpoints share cache entries. |
| 18 | budget metering; seeded BudgetExceeded; attribution | a: HOLDS_WITH_CAVEAT; b: BROKEN-latent (split); c: BROKEN (ledger side only) | yes (18b,18c) | F1, F6 | 18a: pre-call check / post-success charge structurally sound; failed calls leave no ledger row (→ point 20). 18b: `escalate_ocr` swallowed `BudgetExceeded` into REVIEW "verifier failed: BudgetExceeded" — verdict split (see dispositions), recorded BROKEN-latent; fixed by F1. 18c: meta omitted `job_id`/`exam_id` → ledger rows unattributed, per-exam aggregates broken; claimed budget-enforcement degradation REFUTED (per-process BudgetManager, one exam per process); ledger side fixed by F6. |
| 19 | decision traces for the verifier | a: BROKEN; b: BROKEN | yes | F7 | 19a: executed ocr_verify traces lacked cache_hit/usage/request_id/latency and hardcoded `cloud=True` (`reliability.py:383-386`) — fully_local/cache metrics misreport; fixed by F7 for the verifier (grade stages deferred). 19b: attempted-and-failed calls were traced `skipped` with `avoided={'ocr':1,'cloud':1}` (`reliability.py:387-390`) — savings inflation; fixed by F7 (FAILED, no credit). |
| 20 | retry behavior / call accounting | HOLDS_WITH_CAVEAT | yes | deferred | Bounds hold: ≤3 inferences / 9 HTTP attempts per logical call, hard-coded (`base.py:62-63`, `openai_compat.py:89-124,179-220`). Defect: repair-loop tokens escape budget+ledger (`last_usage` overwritten per attempt), `CallResult.retries` always 0, failed calls leave no row — gateway/backend-wide, deferred to next phase. |
| 21 | malformed model output | HOLDS_WITH_CAVEAT | no | F2 (softness) | Fail-closed: local validation regardless of server enforcement, `BackendError` after 3 attempts → REVIEW; only validated objects cached. The minimal-parsable-reply softness (`{"verdict":"supported"}` clearing AUTO) is closed by F2. |
| 22 | provider failure (HTTP/timeout); failed-call accounting | a: HOLDS_WITH_CAVEAT; b: BROKEN | yes (22b) | F7 (trace side) | 22a: no provider failure can become AUTO; advisory path forces REVIEW even on a clean grade (`reliability.py:478-481`). 22b: failed attempt credited as "avoided" cloud call with no ledger row — trace side fixed by F7; failed-call ledger/budget row deferred with point 20. |
| 23 | fail-closed exit enumeration; reachability; seeded claim; empty exit | a: HOLDS_WITH_CAVEAT; b: UNWIRED; c: BROKEN-latent (split); d: BROKEN; e: HOLDS | yes (23b,c,d) | F1 (c), F5 (d) | 23a: seven exits, AUTO reachable only on the two intended (`escalation.py:122-156`). 23b: verifier model call production-unreachable (central finding; not fixed). 23c: budget swallow — confirmed BROKEN by this verifier (split with 18b); fixed F1. 23d: empty-text AUTO exit incl. contradictory empty+`full` fail-open and OCR_OK stamp; fixed F5. 23e: no other unverified-read AUTO path exists. |
| 24 | downstream OCR_UNRESOLVED handling | a: UNWIRED; b: HOLDS_WITH_CAVEAT; c: HOLDS | yes (24a) | — | 24a: evidence-backed withhold branch (`reliability.py:403-414`, grader skipped) is correct but can never fire in production (verify and quality both always None). 24b: heuristic-advisory path grades with score unaffected, forces REVIEW; escalate_grade can still see flagged text on grading trouble; review-UI codes the reason GRADE_UNCERTAIN (label only). 24c: route_item/reviewqueue/apply_review_items never convert OCR trouble to grading work or drop a flag. |
| 25 | transcription immutability; shadow isolation | a: HOLDS; b: HOLDS; c: HOLDS_WITH_CAVEAT | no | — | Single write-once lazy-OCR write point (`reliability.py:350-351`), frozen to extraction.json; no later stage mutates it — `OCRVerifyResult` cannot even represent a corrected reading (`escalation.py:74-79`). Shadow isolation holds today but rests on flag discipline (`_defer_ocr`/`ocr_fn` gating), not object isolation — no deepcopy of the shared extraction. |

## Confirmed defects and dispositions (17 adversarially verified claims)

1. **13a — no crops producer.** Confirmed UNWIRED. The verifier, its quality gate, and its
   budget line are dead code in production; fail-safe but the designed routing cannot occur.
   Disposition: deferred — next-phase wiring (the audit's central finding).
2. **2b — "ONE line/cell" premise / reachability.** Confirmed UNWIRED. No call site can
   supply any image at all; the premise has no enforcement. Disposition: deferred with 13a.
3. **5b — omitted confidence AUTO-eligible.** Confirmed HOLDS_WITH_CAVEAT. Deterministic
   pydantic-schema behavior; minimal reply passed the whole AUTO gate.
   Disposition: fixed (F2, default → `"low"`).
4. **6 — omission handling unreachable.** Confirmed UNWIRED; mechanism fail-closed as
   written. Disposition: deferred — crops root cause (shared with 7, 8, 10, 12b).
5. **9b — partial/illegible with text discarded.** Confirmed BROKEN. The guard requires
   `not text`; legibility is a dead variable afterwards, contradicting the module's own
   routing doctrine. Disposition: fixed (F3, advisory suspicion + persisted legibility).
6. **11 — digit/operator blind spot.** Confirmed UNWIRED (dead `_DIGIT_TOKEN`); verifier
   errata on the `-3` example noted, class stands from len 3 (`x=3`).
   Disposition: fixed (F4).
7. **12a — empty transcription, non-blank crop.** Confirmed BROKEN. AUTO "missing"
   returned before crop lookup even in a crops-supplied path. Disposition: fixed (F5).
8. **18b — seeded BudgetExceeded claim (verifier 1).** Corrected: confirmed=False as
   stated — the "degrades to per-item REVIEW" consequence was mis-derived (the still-tripped
   budget re-raises at the grading stage → PAUSED one stage late, except cache-hit /
   non-counted / no-pack sub-cases), and this verifier reclassified UNWIRED-latent.
   Disposition: kernel real (no re-raise, unlike `escalate_grade`); fixed (F1). See 23c.
9. **18c — budget/ledger attribution.** Corrected: the budget-enforcement consequence
   (cross-exam shared bucket) was REFUTED — BudgetManager is per-process, one exam per
   process — only the ledger-attribution defect stands (rows with `job_id`/`exam_id` None,
   per-exam aggregates broken). Disposition: fixed (F6) for meta; job runner still passes
   no `job_id` into the pipeline.
10. **19a — executed-trace metadata + hardcoded cloud.** Confirmed BROKEN. Cache hits
    traced as non-cached cloud calls; `fully_local` contradicts its own contract.
    Disposition: fixed (F7) for ocr_verify; grade-stage traces (same pattern, hardcoded
    `cloud=True` at `reliability.py:444-452`) deferred — out of verifier scope.
11. **19b — failed call traced as skipped/avoided.** Confirmed BROKEN. `verify=None`
    conflated attempted-failed with never-attempted. Disposition: fixed (F7,
    `attempted`/`error` markers, FAILED trace, no avoided credit).
12. **20 — retry accounting.** Confirmed HOLDS_WITH_CAVEAT (bounds hold; accounting
    defect real and live via grading tasks). Disposition: deferred — backend-wide
    repair-loop usage accumulation + `CallResult.retries` + failed-call ledger rows.
13. **22b — failed-call accounting.** Confirmed BROKEN (same mechanism as 19b plus
    no-ledger-row). Disposition: trace side fixed (F7); gateway-side failed-call
    ledger/budget row deferred with 20.
14. **23b — production reachability.** Confirmed UNWIRED. Every refutation attempted
    (other callers, other names, lazy-OCR crops, other escalate_ocr callers) failed.
    Disposition: deferred — next-phase wiring.
15. **23c — seeded BudgetExceeded claim (verifier 2).** Confirmed BROKEN: the pause
    invariant is deterministically contradicted in the audited path; `escalate_ocr` is
    production-wired, only the crop input is missing. **Split verdict with 18b** (one
    corrected to UNWIRED-latent, one confirmed BROKEN): recorded as **BROKEN-latent** —
    the defect is real in source, unreachable until crops are wired.
    Disposition: fixed (F1, re-raise + PAUSED/BUDGET_PAUSED wrap).
16. **23d — empty-transcription AUTO exit.** Confirmed BROKEN: unsafe legibility default
    (`or "none"`), empty+`full` contradictory fail-open, OCR_OK stamped on an unverified
    read; open questions got a silent zero. Disposition: fixed (F5).
17. **24a — evidence-backed withhold dead.** Confirmed UNWIRED; branch correct in source,
    `evidence_backed` structurally always False in production.
    Disposition: deferred — crops wiring.

## Next phase requirements

1. **Wire the crops.** Produce per-item explanation crops at the extraction/lazy-OCR seam
   (template/survey `explanation_area` regions or the labeled page slices
   `lazy_explanation_ocr` already selects, cropped from the in-memory post-mask `pages`),
   keyed by canonical `(question_id, sub_item_id)`, and pass them into
   `run_reliability_judging(crops=...)` at `cli.py:942` AND into `build_review_items`
   (`webui.py:904`) so human reviewers see the image evidence. Derive `ocr_available`
   from route-configured AND crop-present; add a preflight warning when reliability mode
   runs with `ocr_verify` configured but an empty crops dict.
2. **Benchmark before trusting AUTO.** Run the prepared signal-2 verifier benchmark
   (`scripts/m2_verify_run.py`) — prompt effectiveness, confidence calibration, AUTO-gate
   strictness, and suspicion-signal recall are all UNTESTED_EMPIRICALLY; no AUTO decision
   from the verifier should be trusted before it.
3. **Fix backend retry accounting** (gateway/backend-wide): accumulate usage across
   repair round-trips, set `CallResult.retries`, and record ledger/budget rows for
   failed calls (points 20, 22b, 18a caveat).
4. **Grade-stage trace metadata:** apply the F7 pattern to grade_primary/grading_rag/
   grade_escalate and replace the hardcoded `cloud=True` on grade-stage traces with
   `is_cloud_route` (`reliability.py:444-452`).

## Production wiring status (2026-08-22, pre-API)

Everything around the missing crop producer is now wired and tested
(`autograder/evidencecrops.py`, `tests/test_evidence_crops.py`):

- **Explicit interface** `ExplanationCropProvider.crop(question_id,
  sub_item_id) -> CropResult(status AVAILABLE|UNAVAILABLE, png_b64, reason,
  source, geometry)`; `collect_crops(provider, key)` feeds
  `run_reliability_judging(crops=...)` and records an availability report
  on the run (`ReliabilityRun.evidence_crops`, persisted into
  `result.backend_info["evidence_crops"]`, logged once per exam).
- **Production provider = `UnavailableCropProvider`** — deliberately.
  There is still no calibrated per-question explanation-region geometry
  (`PageRegion` descriptive only; `tablecrop` covers MC rows). Coordinates
  are NOT invented and the full page is NEVER sent as "the crop".
- **Fallback (documented contract):** a suspicious reading with no crop is
  `REVIEW` with reason "suspicious; no evidence crop available (crop
  producer unavailable)" — no verifier call, never AUTO; unsuspicious
  readings proceed to grading unchanged. When a crop IS supplied
  (`StaticCropProvider`, a future calibrated producer) the existing
  deterministic triage, gateway routing (cache, budget, ledger, privacy
  scan), trace stage and typed review reason apply unchanged.
- GUI: the Review queue shows "no image evidence available for this item"
  and the batch-level reason; Advanced shows the provider status.

### Exact status (2026-08-22)

```
BENCHMARK:   READY   (frozen REAL 303 + SYNTHETIC_NEAR_MISS 136; smoke 12; FAR primary)
PRODUCTION:  SAFE BUT UNAVAILABLE
             because calibrated explanation crop geometry does not exist.
             current safe route: suspicious OCR -> no trusted crop -> REVIEW;
             no verifier provider call; no invented coordinates; no full pages
             sent to claim the verifier is wired.
             ExplanationCropProvider stays the extension point for calibrated
             template geometry.
```
