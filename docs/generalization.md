# Generalization & anti-overfitting audit (2026-08-21)

Verdict in one line: the pipeline core is genuinely configuration-driven
(no hardcoded answers, ids, counts, totals, or pixel coordinates in
production paths), but the audit found and removed **10 production-leakage
items** (2 code, 8 prompt), documented 7 suspicious special cases for the
next phase, and re-classified most historical data as no-longer-unseen.
Guard tests: `tests/test_generalization_guards.py`,
`tests/test_claude_independence.py`.

## 0.1 Data classes

- **DEV / REGRESSION** — may be inspected while debugging. Everything
  repeatedly inspected during development IS dev, whatever it was called.
- **CALIBRATION** — used only for selecting thresholds/models/policies.
- **HELD_OUT** — never inspected while modifying the system; one
  final-evaluation use after a configuration freeze. Once its results are
  inspected and acted on, it becomes DEV — permanently.

Split unit is the **exam instance** (one scanned PDF = one student = one
writer); pages are never split across classes.

### Concrete assignment (2026-08-21)

| Class | Contents |
|---|---|
| DEV | `sample_data` (8 audited runs); all of `prob_data` (13 exams, fully benchmarked); grading TRAIN split (25 exams); Stage-A validation exams 002/003/004/007/008 (graded + per-item audited + fixes designed from them); validation exams 009/010/016/017/018 (handwriting feeds HTR/M2 benchmarks that drove the Gemini decision); exam-002 (the consumed "hidden" bench); hebrew_bench v1/v2, m2_grading, all evaluation/ artifacts |
| CALIBRATION | validation exams **019, 029, 035, 040, 041, 042** — never individually run, audited, or annotated. Quarantine rule: threshold/policy/model calibration measurements only; no per-item debugging against them. This is also where the future owner-scored per-answer grading subset should come from. |
| HELD_OUT | `held-out test set/` — 49 PDFs (indices 050, 052, 054-100), committed 2026-07-12, contents never opened; zero references anywhere in code/eval/benchmarks (verified by exhaustive search + MD5 cross-check vs test/: no duplicates) |

**What can no longer honestly be called unseen:** all of DEV above — in
particular the five Stage-A exams (docs/evaluation.md's "only
validation-split results may be quoted as generalization evidence" no
longer holds for them) and the 10 of 16 validation exams whose content
fed benchmarks. Only the 6 CALIBRATION exams and the 49 held-out exams
retain evidentiary value, in their respective roles.

**Held-out caveats:** (a) grade labels are visible in the tracked
filenames (developer-level label exposure; the pipeline never reads them
— accepted residual risk, already in git history); (b) docs say 48 exams,
the directory holds 49 — reconcile with the owner at freeze; (c) no
probability-course held-out set exists at all: prob-template accuracy
claims are dev-set-only until new graded prob exams arrive and are frozen.

**Terminology rule:** "held-out" refers ONLY to this final test set. The
HTR writer-generalization folds must say "left-out writer" — their
"held-out CER" numbers are DEV measurements (two of the "train" writers
are grading-validation exams).

### Held-out final evaluation — exact requirements (do not run now)

1. Finish model selection + integration; freeze configuration per
   docs/evaluation.md (git tag, `backend.describe()`, model digests into
   `datasets/final_test_manifest.json`).
2. Reconcile the 48-vs-49 count; populate the manifest from
   `held-out test set/` via make-manifests; keep `frozen_configuration`
   recorded before first contact.
3. Run `eval-batch` exactly once, masking ON, labels joined post-grading;
   report totals metrics + review rate. No reruns, no per-item debugging;
   any fix derived from the results demotes the set to DEV.

## 0.2 Exam-specific knowledge in production code — findings

**Removed (PRODUCTION_LEAKAGE):**

1. `webui` package discovery hardwired to `prob_data`/`sample_data` →
   now configuration (`GRADER_PACKAGE_DIRS` env / `grader.toml [ui]
   package_dirs`; neutral default `packages/` + `sample_data/`). Owners
   with packages in `prob_data` add one config line.
2. `JUDGE_SYSTEM` asserted a specific historical rubric rule as "the
   exam's own rubric" for every exam → rephrased as a generic principle
   deferring to the per-question rubric.
3-8. Prompt leakage (0.6): the key-parser prompt embedded the real key's
   colour legend ("R,B,G for A1,A2,A3"), two real answer groups (F/F/G,
   A/H/B) with a worked decode, the real version-note answers (3 vs 4),
   the real "accept both A and B" note, and the real Q3 cap structure
   (20x2 cap 36 — also in schema.py's model-visible description); the
   survey/close-read prompts embedded the real student's swap note
   (English with the real question numbers and near-verbatim Hebrew
   "התבלבלתי בין השאלות") and the real instructor score 28/32. All
   replaced with clearly synthetic values; decode/behavior rules kept.
   Pinned by `test_model_visible_prompts_carry_no_current_exam_content`.

**Documented, not changed (SUSPICIOUS_SPECIAL_CASE — next-phase actions):**

- discovery.py phrase-match → `fixed_pages=[1]` auto-template rule
  (currently reachable only via tests; needs lecturer confirmation before
  it may emit structural facts).
- discovery.py auto-emitted 4-column A/B/C/D RTL table + auto-banding
  (should derive from the key's option set; template already makes it
  per-exam config).
- keyrepair.py family-specific header ("שאלה מספר N") + A-I slash-group
  decoding, and cli.py's matching flattening detector (guarded, review-
  flagged; belong in a per-package key-format declaration).
- tablecrop.py mark calibration constants (validated on 130 rows of one
  family; gate banding for NEW families behind a calibration check).
- prompts: flower/petal variant-match criteria; disambiguation prompt's
  hardcoded 4-letter column list; discovery `_SUIT_WORDS` alias table.

**Confirmed absent** (searched, none found in production paths): hardcoded
answer values, student ids, course ids, question numbers/counts, variant
counts, score totals, fixed page counts, absolute pixel coordinates,
dataset-path-keyed branches.

## 0.3 Persistent state taxonomy

| Store | Class | Reset |
|---|---|---|
| gateway request cache (`<state_root>/gateway_cache`) | C disposable | delete dir; content-fingerprinted (no exam id in key) |
| usage ledger (`gateway_ledger/usage.jsonl`) | C accounting | delete file (feeds daily/monthly + campaign cost caps) |
| parsed-key cache (`key_cache/`, global per user) | A course | `--no-key-cache` (read+write), `--key-cache-dir`, delete |
| model-derived alignment cache (`align_*.json`, same dir) | B package | **fixed:** `--no-key-cache` now bypasses the READ too (was write-only) |
| PackStore (`<out>/../packs/<fp>`) | B package | fingerprint auto-invalidates; delete dir |
| decision traces (`decisions.jsonl`) | B exam | unlinked before every re-run |
| review resolutions + apply_to_all log | B exam | note: currently **display-only** — no pipeline re-application exists (docstring corrected; wiring is next-phase UI work) |
| course RAG store (`courses/`) | A course | rebuild index; **fixed:** build-time re-screen + persisted operator overrides (below) |
| job dirs (`jobs/<id>`) | B exam | delete job dir; stage artifacts fingerprint-guarded |
| webui staging dirs (`jobs/_staging-*`) | C | **fixed:** deleted after job creation (original grade-bearing filenames no longer accumulate) |
| VariantCatalogStore / ExactReuseStore / CanaryStore | dormant | zero production callers; semantic grade reuse structurally refused (`reuse.py`) |

No D-class (hidden learned/global exam memory) store exists: no TTL magic,
no store keyed on student identity, no accumulated grades influencing later
grading; variant detection never reads answers/scores.

## 0.4 Fresh-state guarantee

`tests/test_generalization_guards.py::test_fresh_state_rerun_reproduces_the_result`:
a package graded with no prior derived state, then all derived state
deleted (out dir, gateway cache, ledger, packs), reproduces identical
totals and review counts on a fresh run. Model-behavior aspects stay
unrun; the deterministic state boundaries (fingerprints, caches) are what
the test pins.

## 0.5 Course-RAG leakage

Clean by construction where it matters: single ingestion door with
filename + content gates (Hebrew aliases included), zero write-back paths
into the index, retrieval query built only from question/rubric/solution
(never student text — test-pinned), official solution reaches grading via
the pack (sanctioned channel), packs carry full provenance and
auto-invalidate on index rebuild. Two risks found and **fixed**:
`build_index` now re-screens every file in `sources/` (out-of-band files
face the same gates; exclusions recorded in the index manifest), and
operator overrides persist to `course.json` + manifest, so a corpus with
operator-approved flagged material is always distinguishable from a clean
one.

## 0.6 Prompt policy

No production prompt may contain content copied from real exams, keys,
rubrics, student writing, or instructor ink. Examples are synthetic and
labeled as such; per-exam facts travel only through the sanctioned
channels (key parse, pack, template, variants config). Few-shot examples,
if ever added, must record their source and stay disjoint from any set
used to claim accuracy. Enforced by the guard test above.

## 0.7 Evaluation hygiene

See docs/model-selection.md §Hygiene for the per-experiment record fields
(dataset role, config hash, prompt version, model, thresholds,
previously-inspected flag) and the HELD_OUT-becomes-DEV demotion rule.
