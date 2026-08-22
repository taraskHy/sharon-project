# Model selection — candidates, benchmark protocols, budget (2026-08-21)

Status: **prepared, not run.** No OpenRouter key is installed, every cloud
role is `UNSELECTED`, and nothing in this document has spent a token. The
role definitions live in [docs/model-roles.md](model-roles.md); the
candidate sets live in
[evaluation/model_selection/candidates.toml](../evaluation/model_selection/candidates.toml).

Rules that bind every experiment below:

1. Winners are chosen **empirically per role** — never by preference. Model
   slugs and prices are mutable external data (config), never grading logic.
2. **No benchmark script may bypass ModelGateway.** Cloud calls route
   through task routes in a benchmark `models.toml`; harnesses that predate
   the gateway (`scripts/m2_bench_run.py` direct adapters) must gain a
   gateway-routed adapter before any cloud candidate run.
3. Every run happens under the campaign budget (below) with a per-run cost
   report, and is recorded with the hygiene fields of §Hygiene.

## Role → candidate map

`UNSELECTED` is expressed in `models.toml` as `model = "UNSELECTED"` (loads
fine; any run/benchmark needing the role refuses with a pointer to the
candidate registry) or by leaving the task's `${ENV}` slug unset (whole
config refuses at load). Reliability/shadow grading refuses at RUN level
when `grade_primary` is not usable (`reliability.py` gate) — never one
silent REVIEW per item.

| Role | task | candidates (order = none) | benchmark |
|---|---|---|---|
| OCR_PRIMARY | `ocr_primary` | gemini-3.7-flash, gpt-5.6-luna, claude-sonnet-5 | B1 |
| OCR_VERIFY | `ocr_verify` | gpt-5.6-luna, claude-sonnet-5, gemini-3.7-flash | B2 |
| GRADE_PRIMARY | `grade_primary` | gpt-5.6-luna-pro, gemini-3.7-flash, claude-sonnet-5 | B3 |
| GRADE_ESCALATE | `grade_escalate` | gpt-5.6-sol, claude-opus-5, claude-sonnet-5 | B4 |
| MC_RESOLVE_CLOUD | `mc_resolve_cloud` | gpt-5.6-luna, gemini-3.7-flash | B5a |
| VARIANT_RESOLVE | `variant_resolve_cloud` | gpt-5.6-luna, gemini-3.7-flash | B5b |
| ALIGN_RESOLVE | `align_resolve_cloud` | gpt-5.6-luna-pro, gemini-3.7-flash | B5c |
| MC_RESOLVE_LOCAL | `mc_resolve` | local Ollama (kept) | B5a (baseline) |
| RAG embedding | — | local bge-m3 (kept) | not in the race |

## Reference ground truth — the 129 vs 102 accounting

`autograder bench references` prints the explicit breakdown (frozen hashes:
audit `eace95fc…`, items `25463e91…`, references `4a93e826…`):

```
129 total
├── handwritten (manual audit required): 102   (86 handwritten_line + 16 handwritten_cell)
│   ├── confirmed: 69   provenance class audited_confirmed
│   ├── corrected: 33   provenance class audited_corrected (audited text used)
│   └── ambiguous: 0
└── other categories (text layer, no manual audit): 27
    ├── printed_rtl 8, formula_printed 7, mixed_he_en 7, option_row_association 5
    └── provenance class text_layer_mechanical: born-digital PDFs; the reference
        is the embedded text layer (objective, mechanical provenance recorded per
        item in references.json) — never model output, never a manual
        transcription; outside the manual audit scope by rule
```

Deterministic gate: `benchmark.manifests.validate_reference_provenance` —
every item participating in strict OCR scoring must carry one of
`audited_confirmed | audited_corrected | text_layer_mechanical`; the runner
refuses any other class (unchecked, ambiguous, unknown provenance) and the
adapter never scores against it. No unaudited historical reference is used
silently.

## B1 — OCR benchmark

Frozen protocol preserved: `evaluation/hebrew_bench_v2` (129 items, 102
owner-verified + 27 born-digital references; references read only
post-inference), run via `scripts/m2_bench_run.py --config-id <arm>` and
scored by `scripts/m2_bench_eval.py` into `evaluation/m2_bench_results.csv`
(31 frozen rows, 9 configs already comparable). One pre-registered arm per
candidate (`arm_freeze.md` convention), identical images/preproc/prompt.

Metrics (per category and overall): CER, **median CER**, usable ≤ 0.25,
usable ≤ 0.50, omission rate (deletions/GT words), semantic
substitution/addition risk (hallucination-rate + hard-cell honesty + the
`gemini_rag_fidelity_audit.md` image-grounded audit protocol for changed
items), schema failures, latency, input/output tokens, actual provider
cost (ledger `reported_cost`).

Cost shape: ~129 calls x 1 run per candidate; crops only. Gateway-routed
adapter required first (rule 2).

## B2 — OCR-verifier benchmark

Input per item: **crop image + frozen OCR transcription. Nothing else** —
no reference, no CER, no rubric, no official answer, no course material,
no grader output (this is the verifier's production contract; see
docs/ocr-verifier-audit.md). Protocol basis: the committed-but-never-run
signal-2 campaign `scripts/m2_verify_run.py` (a-priori thresholds T1-T4,
outputs under `hebrew_bench_v2/outputs/<arm>_verify/`), analysis
`scripts/m2_verify_analysis.py`.

Label plan: the existing join target (fixed-judge preserved/silent labels
in `evaluation/m2_grading/*.jsonl`) is a PROXY, not truth. Before any
model is selected for `ocr_verify`, the owner audits a labeled set of
(crop, transcription) → supported / review verdicts (start: the 23
gemini3_flash grading cells + deliberately corrupted transcriptions as
known-review cases).

Metrics: true-supported, false-supported (**dangerous false accepts —
the headline metric**), true-review, false-review, false-accept rate,
false-reject rate, schema failures, latency, cost. A model that
false-accepts corrupted transcriptions is disqualified regardless of its
other numbers.

**Dataset layers (after the manual audit froze, 2026-08-22):**

- *Raw pool* — `verifier_bench/` flat emission (690 cases = every
  persisted historical OCR output for the 102 audited items; 9
  supported / 681 review). Kept byte-identical; NOT a selection benchmark
  (98.7% negative, heavily correlated outputs per image).
- *Selected benchmark* — built by `scripts/verifier_select.py propose`
  and, after owner approval of the split, `freeze` →
  `verifier_bench/selected/`. Composition: exactly one POSITIVE per
  audited item (crop + audited reference as the candidate, expected
  SUPPORTED; model-visible row indistinguishable from a negative) and real
  historical error NEGATIVES deduplicated per image (canonical-normalized
  text + digit/sign signature), at most 2 per image chosen for coverage
  (subtle low-distance error first — the dangerous false-accept class —
  then the candidate adding the most new error kinds; documented
  exception: a 3rd when it is the image's only number/sign error).
  Multi-label error kinds preserved. Writer-level DEV / CALIBRATION /
  HELD_OUT split; every case of one image lives in one split (zero image
  overlap by construction).

- *Frozen 2026-08-22 (Split A):* 102 positives + 201 real negatives = 303
  cases; DEV = e002, e003, e007 (33 images); CALIBRATION = e004 (23);
  HELD_OUT = e005, e006 (46). Decision, rationale, writer assignment,
  image ids per split, raw-pool hashes and the zero-overlap assertion are
  persisted in `verifier_bench/selected/manifest.json` (+ `CHECKSUMS.sha256`).
- *SYNTHETIC_NEAR_MISS* (second component, `scripts/verifier_synth.py`,
  proposal only until approved) — the real pool has only 9 subtle
  negatives, so this layer tests small grading-relevant errors: deterministic
  generic OCR-fidelity corruptions of the frozen audited references (digit
  substitution, operator substitution/removal, decimal-point corruption,
  super/subscript loss, one-character deletion, short-token omission, short
  token duplication), rules fixed before any model output
  (`RULES_VERSION`); selection policy v2: exactly one TEXT case per image
  chosen by a deterministic hash rotation over the three text rules (so
  universally-applicable char deletion cannot dominate) plus at most one
  NUMERIC case where the reference contains numeric/math material — max 2
  per image; each case inherits its image's split from the frozen REAL
  manifest. Never "corrected answers". Reported SEPARATELY from REAL:
  REAL → FAR, FRR, SUPPORTED precision, REVIEW rate; SYNTHETIC → FAR
  overall, FAR by corruption type, FAR on the numeric group; COMBINED only
  secondarily, never hiding either source.

- *SYNTHETIC_NEAR_MISS frozen 2026-08-22* (owner-approved, selection-policy-v2):
  136 cases = 102 text + 34 numeric/math (short_token_omission 39,
  token_duplication 34, char_deletion 29, digit_substitution 26,
  operator_substitution 8); DEV 44 / CALIBRATION 27 / HELD_OUT 65;
  `verifier_bench/synthetic/` with manifest (rules + policy versions, source
  audit hash, REAL benchmark hashes, split assignment, image + case ids per
  split, composition, zero-overlap assertion) and `CHECKSUMS.sha256`;
  `verifier_synth.py verify` re-checks every invariant.

**Model-selection objective for OCR_VERIFY — safety first.** Primary
metric: **FALSE ACCEPT RATE** = incorrect transcription classified
SUPPORTED (on negatives). Also reported: false reject rate (correct
transcription sent to REVIEW), SUPPORTED precision, REVIEW rate, schema
failure rate, cost, latency. Models are NOT chosen on overall accuracy:
with a balanced set a high accuracy can hide a fatal false-accept rate on
subtle number/sign errors. Selection order: lowest FAR within budget, then
REVIEW rate (review cost), then schema failures/latency/cost.

## B3 — grade-primary benchmark

Frozen OCR transcriptions only — **zero OCR calls**. Constant across
candidates: question pack (`QuestionGradingPack`, frozen key fingerprint
`0758cd7f…` convention from `scripts/grading_rag_ab.py`), rubric, official
solution, selected option, transcription, prompt (`GRADE_SYSTEM`,
`prompt_version` pinned), schema (`GradeResult`), decoding config
(temperature 0, same max_tokens).

Label reality (docs/datasets.md): **no per-item instructor labels exist
anywhere.** Until an owner-scored per-answer subset exists (create it from
the CALIBRATION tier — see docs/generalization.md), grading benchmarks can
measure only (a) decision preservation vs the frozen fixed-judge verdicts
(NOT accuracy — must be labeled as such) and (b) exam-TOTAL error via
`autograder eval-batch` (metrics.py: exact, ±2/±5/±10, MAE, median AE,
RMSE, signed error, review rate) with labels joined post-grading.

Metrics once per-item labels exist: exact score accuracy, mean absolute
score error, harmful upgrades (score above labeled), harmful downgrades,
rubric decision correctness (per rubric item), evidence validation
failures (`evidence.py`), invalid schema, AUTO rate, REVIEW rate, tokens,
calls, cost, latency.

## B4 — grade-escalation benchmark

Evaluate **only cases that genuinely require escalation**: harvest from B3
runs the items where the primary was unclean (validation failure /
uncertainty / disagreement — `GradeDecision.stage == "escalated"`
triggers), plus the 5 pre-registered G-cells in `grading_rag_ab.py`. Do
not benchmark expensive models across easy answers. Metrics as B3 plus:
resolution rate (unclean → clean-consistent), disagreement-with-primary
distribution, added cost per resolved item.

## B5 — MC / variant / alignment benchmarks

Audited datasets that exist today:

- **B5a MC:** `evaluation/prob/manual_audit.json` — 130 audited rows
  (agent-audited, owner-verified where totals disputed) + the
  deterministically-AMBIGUOUS-row harness `scripts/mc_fallback_bench.py`
  (gateway-routed, post-hoc label join). The frozen local result (10
  ambiguous rows, 1 resolved) is too small — expand the ambiguous-row set
  from further audited jobs before selecting a cloud resolver.
- **B5b variant:** 13/13 audited suit labels (prob) + 5/5 Stage-A flowers;
  build the standalone labeled cover-crop set (3 variants x N covers from
  existing scans) so variant reads are testable without full pipeline runs.
- **B5c alignment:** operator-verified mappings
  `sample_data/Exam_solution.alignment.json` (A1/A2/A3) + frozen
  before-fix artifacts — ready ground truth no harness exploits yet; score
  `model_align_question` output against them.

Metrics for all three: exact correctness, unsafe/wrong AUTOMATIC
resolution (the veto metric — a wrong automatic variant/alignment corrupts
every downstream mapping), abstention rate (unresolved → escalate/review
is acceptable behavior, not failure), cost.

## Budget — the $10 campaign ceiling (Part D)

Configuration (prepared; no key installed, `/api/v1/key` never called):

- `[budget] max_cost_total = 10.00`, `soft_fraction = 0.8` → hard stop at
  $10.00 cumulative reported cost, warning from $8.00. Enforced by
  `BudgetManager` against the **persisted ledger**, so it survives across
  runs and processes.
- **One shared state root for the whole campaign** —
  `evaluation/model_selection/state/` — so every run writes one ledger and
  the ceiling is global. A run using a different state root escapes the
  ceiling; don't.
- **Predicted-cost refusal:** `gateway.call` estimates the upcoming call
  (payload size + route max_tokens x `[pricing]`) and refuses the call
  that WOULD cross the ceiling. Every cloud candidate must therefore have
  a `[pricing."vendor/slug"]` entry (estimator-only, local, never fetched)
  before its first run.
- Ledgers kept: (1) the local usage ledger (per-call tokens, provider,
  request id, reported cost; `autograder.spend.ledger_summary` gives
  per-task / per-model / cumulative); (2) once a key is installed, OpenRouter's
  own key-usage metadata (GET /api/v1/key) can be fetched on demand
  (`OpenRouterBackend.key_metadata()` / `fetch_key_metadata`, secret-free
  parse) and is shown NEXT TO the local ledger in the GUI's Advanced screen —
  supported now, deliberately never called automatically and not called yet.
- Every run must produce `usage.run_cost_report(ledger, baseline_rows)`:
  cost before / run cost / cost after, calls + tokens by model and by
  task. Attach it to the run record.

Execution order when the owner installs the key (2026-08-22 decision —
**OCR_PRIMARY before OCR_VERIFY**, smoke subsets first):

1. `scripts/openrouter_smoke.py` (exactly 2 paid calls; routing / ledger /
   budget / secret hygiene)
2. OCR_PRIMARY smoke — `bench run --role ocr_primary --split dev --subset smoke`
   (8 pre-registered DEV cases) per candidate
3. OCR_PRIMARY DEV (60 cases) per surviving candidate
4. OCR_PRIMARY CALIBRATION finalists (23 cases) → owner reads the report, writes
   the winner into models.toml by hand
5. OCR_VERIFY smoke — `--subset smoke` (12 pre-registered DEV cases: REAL
   supported / omission / substitution / addition / number-sign / subtle +
   SYNTHETIC digit / omission / deletion / duplication)
6. OCR_VERIFY DEV (99 REAL + 44 SYNTHETIC, reported separately)
7. OCR_VERIFY CALIBRATION finalists (67 + 27)
8. **selected-OCR compatibility test**: take the actual errors produced by the
   selected OCR_PRIMARY on DEV/CALIBRATION (its transcriptions vs the audited
   references), build candidate/crop pairs from them, and test whether the
   selected OCR_VERIFY catches them safely (FAR on those real errors). The
   frozen REAL + SYNTHETIC verifier benchmark stays as it is; this is an
   additional, later compatibility check.
Then B5 (MC / variant / alignment as their datasets become READY), B3, B4
under the same $10 ceiling. HELD_OUT is touched only by `bench final-eval`.

## Benchmark harness (2026-08-22, pre-API)

One provider-independent runner for every cloud role lives in
`autograder/benchmark/` and is driven by `autograder bench ...`:

| command | what it does |
|---|---|
| `bench list` | roles, frozen-dataset status (hash-verified), split counts, candidates, held-out log |
| `bench inspect --role R [--preview]` | manifest summary; DEV request preview (CALIBRATION/HELD_OUT previews refused) |
| `bench dry-run --role R --split S --candidate SLUG` | builds every request, runs the leakage guard, predicts cost from the local `[pricing]` table, writes `plan.json` — **zero provider calls** |
| `bench run ... --i-understand-this-spends-money` | live run through ModelGateway (cache, ledger, privacy scan, $8/$10 budget) |
| `bench report --run-dir DIR` / `--role R --split S` / `--historical` | one run / all runs / historical OCR outputs re-scored on the audited references |
| `bench compare --role R --split S` | candidates side by side — **no winner is selected** |
| `bench held-out-log` | the permanent record of HELD_OUT executions |

Architecture: `manifests.py` (frozen manifests: hash verification, split /
component selection; REAL and SYNTHETIC are separate components) ->
`registry.py` (candidates.toml as data) -> `roles.py` (per-role adapter =
production prompt + schema + scoring; `model_visible_fields` whitelist) ->
`runner.py` (resume, raw `outputs.jsonl`, `run.json` with candidate / route
fingerprint / prompt sha256 / schema sha256 / adapter version / manifest
hashes / git commit, `metrics.json`, `usage.json` via `run_cost_report`) ->
`report.py`. Benchmark routes set `validation_retries=0`: a malformed
answer is a schema failure, never silently repaired. The run directory is
keyed by the configuration hash, so a changed config is a different run.

Dataset status: `ocr_verify` FROZEN (REAL 303 + SYNTHETIC 136),
`ocr_primary` FROZEN (129 items; references via
`reference_for_scoring(mode="final")`; writer split = Split A, text-layer
items DEV), `grade_primary` / `grade_escalate` / `mc_resolve_cloud` /
`variant_resolve` / `align_resolve` DECLARED, NOT BUILT (generic frozen
format under `evaluation/model_selection/datasets/<role>/`, writer
`benchmark/datasets.py::write_declared_dataset`; no dataset is fabricated).
B3 label reality is unchanged: no per-item owner labels exist, so the grade
adapter reports accuracy metrics as *unavailable* until they do.

### Pre-registered smoke subsets

`evaluation/model_selection/smoke/<role>_smoke.json` — frozen BEFORE any new
model output by deterministic rules (`benchmark/smoke.py`, rules
`smoke-rules-v1`: each slot takes the smallest qualifying DEV case id).
OCR_PRIMARY_SMOKE = 8 cases (ordinary line, difficult handwriting, short
answer, longer answer, formula/numeric, mixed Hebrew/English, numeric
handwritten cell, option-row association); OCR_VERIFY_SMOKE = 12 cases
(3 REAL supported, REAL omission / substitution / unsupported addition /
number-sign / subtle, SYNTHETIC digit substitution / short-token omission /
char deletion / token duplication). Each file records the slot and rule per
case and a `selection_sha256`; `bench run/dry-run --subset smoke` loads it,
re-verifies the hash, refuses non-DEV splits, and never re-selects. The
subsets are never optimized after results exist.

### Dataset status per role (2026-08-22, `autograder readiness`)

| role | status | dataset | labels | notes |
|---|---|---|---|---|
| OCR_PRIMARY | **READY** | frozen hebrew_bench_v2, 129 items (DEV 60 / CALIB 23 / HELD_OUT 46) | audited refs (102) + text-layer (27), all provenance-valid | smoke 8 |
| OCR_VERIFY | **READY** | frozen REAL 303 + SYNTHETIC 136 | expected verdict per case | smoke 12; REAL/SYNTHETIC separate |
| GRADE_PRIMARY | **NEEDS_OWNER_LABELS** | `evaluation/model_selection/datasets/grade_primary` — 67 cases (DEV 32 / CALIB 14 / HELD_OUT 21; writer Split A); 13 incomplete cells excluded | 0 owner labels; `owner_labels.json` via `scripts/grade_label_ui.py` | inputs = NO-RAG packs from the frozen key + audited transcriptions; runnable now for decision/validation metrics only |
| GRADE_ESCALATE | **PENDING_OTHER_EXPERIMENT** | harvested by `bench build-escalation --from-run <grade_primary run>` | owner labels of the harvested cases | difficulty is never invented |
| MC_RESOLVE | **READY** (DEV only, n=10) | `datasets/mc_resolve_cloud` — deterministic band crops of the 10 ambiguous prob rows | agent-audited answers (provenance on every label; owner verified 3 totals only) | expand before selecting a cloud resolver |
| VARIANT_RESOLVE | **READY** (DEV only, n=16) | `datasets/variant_resolve` — marker-region crops (page-1 bottom third): 13 prob suits + 3 Stage-A flowers | 13 agent-audited suits (10 pinned by totals) + 3 operator content-verified flowers | production sends the full page; parity run can render locally |
| ALIGN_RESOLVE | **NOT_AVAILABLE** | — | operator mapping exists (`sample_data/Exam_solution.alignment.json`) | printed booklet texts need an OCR pass (image scans, no text layer); canonical key artifact has a duplicate-prompt defect (Q3.9) |

Builders: `bench build-grading | build-mc | build-variant | build-escalation | build-align`
(`benchmark/datasets.py`; local deterministic processing; each refuses to overwrite). Owner
labels: `python -m streamlit run scripts/grade_label_ui.py` writes
`owner_labels.json` next to (never inside) the frozen dataset; `bench owner-labels
--role grade_primary` reports the remaining count; the manifest merges confirmed
labels and hashes the file into the run identity.

### Spend safety — the live-call sequence (Part 7)

```
OPENROUTER_API_KEY exists -> GET /api/v1/key (explicit) -> record starting key usage
  (+ local ledger at that instant) -> compare key delta vs ledger delta (show BOTH)
  -> budget safe? (local cumulative + predicted < $10) -> call allowed
```
`spend.campaign_preflight` runs once per live run before any provider request
(`runner._live_preflight`; cloud routes only). **The local ledger controls the
$10 ceiling**; the account-side number is informational, and a disagreement
(> $0.05 between the two deltas) refuses the run until the owner looks. The
per-call BudgetManager check still guards every subsequent call (predicted cost
from the local pricing table; the call that would cross $10 is refused before it
starts; warning from $8).

### Split discipline (Part 3)

- DEV may be inspected; CALIBRATION selects; HELD_OUT is reserved.
- HELD_OUT **cannot be dry-run or previewed at all**: `bench dry-run --split
  held_out`, `bench run --split held_out` and `bench inspect --split held_out
  --preview` all refuse ("HELD_OUT is reserved for final evaluation and cannot
  be previewed/dry-run"). The ONLY path that executes it is
  `bench final-eval --role R --candidate SLUG --confirm-held-out
  --i-understand-this-spends-money` (live). Every final evaluation is appended
  permanently to `evaluation/model_selection/HELD_OUT_EXECUTIONS.jsonl` with
  role, run id, git commit, config hash, model, prompt/schema sha256, adapter
  version and manifest hashes — written only once the run can actually
  execute (credential + readiness checks passed). Once held-out results are
  inspected and used to change anything, the split is **demoted to DEV**
  (record it in docs/generalization.md) and can never again be treated as
  unseen. Held-out labels/predictions are never shown by preview tools.
- OCR_VERIFY: REAL and SYNTHETIC are always reported separately
  (`metrics.REAL`, `metrics.SYNTHETIC` with by-corruption-type and numeric
  FAR); `COMBINED_secondary` exists only as a secondary figure.

### Verifier contract (Part 4)

The verifier request is exactly production's (`OCR_VERIFY_SYSTEM`, one
image block, "Proposed transcription: ..."): crop + candidate, nothing else.
`runner.leakage_check` refuses any request that carries a label value, a
label field name or a split name, and any input outside the adapter's
whitelist; the acceptance gate scored is production's AUTO gate
(supported AND high/medium AND no reported omissions/substitutions/additions).
Primary metric: FALSE ACCEPT RATE; also FRR, SUPPORTED precision, REVIEW
rate, schema-failure rate, latency, tokens, reported cost.

## Hygiene (0.7)

Every experiment records: dataset role (**DEV / CALIBRATION / HELD_OUT**),
config hash, prompt version, model slug, thresholds/policy, and whether
results from this dataset were inspected before ("previously_inspected").
Rules:

- Never tune a threshold/model/prompt and report performance on the same
  data as unbiased final performance.
- Once HELD_OUT results are inspected and used to change the system, that
  dataset **becomes DEV** for every subsequent iteration — record the
  demotion in docs/generalization.md.
- The word "held-out" is reserved for the final test set
  (`held-out test set/`); HTR writer-fold vocabulary must say "left-out
  writer" (see docs/generalization.md §terminology).
- Pre-register arms before results exist (the `arm_freeze.md` /
  `m2_verify_run.py` convention); frozen experiment artifacts are never
  rewritten.

## Prerequisites checklist for the first paid run

1. `OPENROUTER_API_KEY` set (never in a file); candidate env slugs set for
   the arm under test.
2. Campaign `models.toml` with `[budget] max_cost_total = 10.00` and
   `[pricing]` entries for every candidate; state root
   `evaluation/model_selection/state/`.
3. `openrouter_smoke.py` green.
4. `autograder bench dry-run` green for the arm (leakage check passed,
   predicted cost shown) — the harness is gateway-routed for every role.
5. Owner approval for the specific run, recorded with its hygiene fields.
