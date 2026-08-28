# Overnight report — local grader output contract (2026-08-28, ~03:27–04:30)

Bounded mission completed and STOPPED. Ground truth everywhere: actual
instructor-assigned grades + actual selection correctness + frozen production
scoring policy; A/B/C/D audit decisions were flags only; no target reached any
model request. Every number below was independently recomputed from raw
artifacts by zero-inference reviewers.

## A. Baseline

- Starting HEAD `3412f95` (clean tree, branch `initial-prototype`).
- Dataset hashes verified against manifest + both freezes: inputs `61ea571e…`,
  labels `2281f063…`, manifest `f3caddb9…`, final_labels `3bdd38c3…`.
- Historical FullDev audit recomputed from persisted artifacts before any
  implementation: **agrees on every checked quantity** (26 outputs, 26×
  `rubric_items=[]`, 25 failures, 25 with a misplaced verbatim span, 20 over
  200 chars, 18/26 verdict-exact, single AUTO = e002_q2_r4, 24/32 end-to-end).
- Machine: RTX 2000 Ada 15.4 GB, 63.4 GB RAM; candidates installed
  (8B digest `0533d74300e4`, 30B `c871fc73fabc`); HELD_OUT log absent.

## B. Implementation

- **Prompt** `grade-v4-charitable-local` = grade-v4-charitable **verbatim** +
  mechanical OUTPUT CONTRACT (rubric_items is the only evidence channel; one
  exact ≤200-char substring per credited item; met=false span grounds a zero;
  top-level evidence null; uncertain=true instead of fabricating). Prompt sha
  `d8c799d1bdec…`, schema sha `2a19252fd837…`, experiment sha
  `e86845b9c85f…` (freeze); v3/v4 prompts untouched.
- **Validation** `grade-validation-v2` (shared production + benchmark path):
  ungrounded invalid verdict on non-empty text → REVIEW (typed
  `ungrounded_invalid_verdict`); the PRODUCTION verdict — not `score>0` —
  selects the grounding rule (closes the (0, 0.001·max] epsilon seam);
  met=true on a zero verdict = contradiction; unknown `evidence_policy`
  fails closed; unknown/duplicate ids checked over every entry; legacy-vs-
  structured met contradictions refused; spans < 3 normalized chars never
  verify (tightening; the Hebrew normalizer itself untouched); transcription
  None = unverifiable (≠ blank). Blank/whitespace answers unchanged.
- **Schema**: GradeResult + RubricItemGrade `extra="forbid"`; adapter
  `grade-bench-v3` converts schema-rejected outputs into schema-failure rows.
- **Runner**: `-Calibration` + `-PromptVersion` (defaults from the active
  freeze); smoke gate now requires a LIVE failure-free smoke of the same
  candidate AND prompt version. Split restricted to dev|calibration.
- Files: autograder/{escalation,evidence}.py, autograder/benchmark/roles.py,
  scripts/{run_local_grade_primary.ps1,local_grade_freeze.py,
  replay_structural_audit.py}, candidates.toml, 6 test files.

## C. Tests and review

- Targeted + full suite before inference: **1358 passed, 0 failed** (3
  unrelated skips). Re-run after the post-inference rounding fix: 1358 again.
- Pre-inference adversarial review (4 agents): 23 findings; every confirmed
  generic defect fixed (epsilon seam, policy fail-open, met=true zero
  laundering, trivial-span vacuity → 3-char minimum, dry-run smoke gate,
  one-sided prompt gate, legacy contradictions, adapter crash, freeze
  HELD_OUT guard gap). Residual documented limitation: a short real word
  (stopword) still verifies — the gate is anti-fabrication, not semantic
  relevance.
- Post-inference verification (4 agents): all metrics reproduce from raw
  artifacts (537 checks; one 1e-4 balanced-accuracy rounding artifact found
  and fixed at source, artifacts regenerated); 28/28 spans re-verified,
  0 wrong-source, all AUTO decisions re-validated; resources reproduce
  exactly; methodology clean (leakage_check on all rebuilt requests: zero).
- Implementation commits (pushed before any inference): `a37a571`,
  `2776d47`, freeze `ac3227a`.

## D. Historical replay (counterfactual — NOT actual model performance)

Of the 25 FullDev evidence failures: **16/25** would have verified with a
span the model itself isolated (5 clean whole-field + 11 quote-in-commentary);
still failing without new copying: 8 paraphrase-only, 1 wrong-source; 25/25
carried a 12–200-char verbatim span somewhere in the misplaced field; 20/25
exceeded 200 chars as submitted; the run's only zero-verdict AUTO (a harmful
undergrade) now routes to REVIEW. Artifacts: REPLAY_STRUCTURAL_2026-08-28.*.

## E. Structural DEV smoke (2 frozen cases each; no quality claim)

| Model | Cases | Schema pass | Correct evidence structure | Evidence valid | Unsafe zero AUTO | Latency | Resources |
|---|---|---|---|---|---|---|---|
| qwen3-vl:8b-instruct | 2/2 | 2/2 | 2/2 (rubric_items, evidence null) | 2/2 spans verify | 0 | 4.7s / 2.5s | 9.9 GB VRAM, 100% GPU |
| qwen3-vl:30b-a3b-instruct | 2/2 | 2/2 | 2/2 (incl. a GROUNDED met=false zero) | 2/2 spans verify | 0 | 47.2s cold / 3.1s | 20 GB, 37%/63% CPU/GPU offload |

**Both PASS** → both ran CALIBRATION. The FullDev failure mode (0/26
rubric_items) is fully reversed by the contract.

## F. CALIBRATION — local explanation-grader quality (strict 11: 7 valid, 4 partially_valid)

| Model | Strict | Verdict acc | Macro-F1 | Balanced | Partial recall | Valid recall | Up | Down | AUTO | REVIEW | Evid. fail | Latency (med/max) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3-vl:8b-instruct | 11 | **63.64%** (7/11) | **0.7179** | **0.6071** | 0.50 | 0.714 | 1 | 3 | 100% | 0% | 0% | 2.12s / 11.6s |
| qwen3-vl:30b-a3b-instruct | 11 | 27.27% (3/11) | 0.325 | 0.3214 | 0.50 | 0.143 | 0 | **8** | 100% | 0% | 0% | 2.76s / 21.4s |

Baselines (strict 11): always-valid **63.64%** / balanced 0.5 / macro-F1
0.3889; always-partially_valid 36.36% / 0.5 / 0.2667; majority 63.64%.
Uncertainty 0%, schema failures 0%, evidence engagement 100% for both.

## G. End-to-end instructor agreement (strict 11)

| Model | Exact final score | MAE | Overgrades | Undergrades |
|---|---|---|---|---|
| qwen3-vl:8b-instruct | 7/11 (63.64%) | 1.0909 | 1 | 3 |
| qwen3-vl:30b-a3b-instruct | 3/11 (27.27%) | 1.8182 | 0 | 8 |

(No actual-0 row exists: the two zero-score CALIBRATION cases have unresolved
selection correctness and sit outside the frozen derivable subset.)

## H. Per-case results

Full per-case tables (actual score, derived verdict, prediction, spans,
validation, decision, audit flag, strict inclusion) are in
`CALIBRATION_CONTRACT_2026-08-28.{json,md}`. Highlights: both models fail the
same two cases (e004_q1_r1 — the case whose committed blind-audit decision B
says instructor practice is more lenient than the literal rubric — and
e004_q2_r6); only-8B-correct: 6 cases; only-30B-correct: 2.
**Diagnostic e004_q2_r8** (audit C, excluded from strict): actual 2.0, derived
target partially_valid, 8B raw 0.0 → invalid, 30B likewise; instructor grade
and target preserved, nothing relabelled.

## I. Resources

8B: 100% GPU (9.9 GB), calibration 12 cases in ~36s live time, 20,575 in /
773 out tokens. 30B: 20 GB with 37%/63% CPU/GPU offload, cold load 47s,
median 2.76s, max 21.4s, 644 out tokens; no OOM, no thrashing (36 GB RAM
free during the run). Machine profiles recorded per execution (note: they are
BOM-prefixed JSON; use utf-8-sig).

## J. Model decision (no automatic winner; registry stays UNSELECTED)

- **qwen3-vl:8b-instruct — MAYBE.** The contract fully fixed the output
  channel (engagement 100%, failures 0%, AUTO 100%, ~2s). Judgement beats
  both trivial baselines on balanced accuracy and macro-F1 (0.718 vs 0.389)
  but only TIES always-valid on raw accuracy, and 3 harmful undergrades now
  AUTO-finalize because validation passes. **Decision needed from the owner:**
  the deterministic safety knob — AUTO only `valid` verdicts, route
  partial/invalid to REVIEW — would cap undergrade harm at 0 for ~36% review
  on this population. Not implemented (post-CALIBRATION change).
- **qwen3-vl:30b-a3b-instruct — DROP for GRADE_PRIMARY.** 0 upgrades /
  8 harmful downgrades of 11 (the qwen 27B harsh-downgrader pathology),
  below every baseline, plus offload latency. Structure was perfect; the
  judgement is not.
- Exact next steps (zero-cost): owner reads this report + decides the
  verdict-gated-AUTO knob and whether the q1_r1/q2_r6 misses warrant a
  rubric-text review (both carry committed audit flags). Next inference
  (owner-gated): none required for the 8B contract itself; a laptop parity
  smoke or the verdict-gate A/B on CALIBRATION would each be a new
  pre-registered experiment.

## K. Limitations

- **invalid-class performance = NOT MEASURED** (no authoritative invalid
  example exists in any split).
- CALIBRATION is ONE writer (e004); nothing here is production readiness.
- Strict denominator excludes the audit-C evidence-issue case.
- No HELD_OUT, no RAG, no other grading policies/rubric families tested.
- A short real word still verifies as a span (documented residual); 7/24
  calibration spans verify via the normalizer (whitespace/Unicode), not
  byte-exact placement — by design of the frozen matcher.

## L. Git

Commits (all pushed to origin/initial-prototype): `a37a571` (contract +
validation), `2776d47` (runner/registry/freeze machinery/replay), `ac3227a`
(pre-registration freeze), plus the post-run artifact commit that carries
this report. Secret/model-file/DB scans clean on every commit; models.toml
(gitignored) untouched tonight; no HELD_OUT content anywhere; final tracked
tree clean.

## M. Confirmations

- OpenRouter calls = **0**; cloud grading calls = **0**; OpenRouter OCR = **0**;
  OCR calls of any kind = **0**; RAG/embedding calls = **0**; HELD_OUT calls = **0**.
- Local inference evaluations = **28 exactly** (2+2 smoke, 12+12 CALIBRATION),
  confirmed by the gateway ledger (28 entries, all task=grade_primary,
  backend ollama, cloud=false) — the authorized maximum was 4+24.
- qwen3.8:27b-q4_K_M executed **0** times (dropped; registry refuses it).
- Cloud spend = **$0**. Instructor grades modified = **0**. A/B/C/D decisions
  used as target labels = **0**. Live labeling DB trio (labels.db/-shm/-wal):
  sha256-verified byte-identical to the start-of-night snapshot.
- Environmental note (not this session's doing): sporadic external
  `/api/chat` calls to the local 8B appear in the Ollama server log
  (02:56–04:07, ~1–8s each) from another localhost client on this machine;
  controlled re-runs proved the repo's pytest/freeze/preflight paths make
  zero model calls. Worth identifying the client, but harmless to the
  frozen experiment (local-only, different context window, no repo writes).
