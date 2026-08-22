# Manual reference audit — hebrew_bench_v2 (2026-08-21)

Before any model benchmark or API-key installation, the manually
transcribed handwriting references of the frozen benchmark are validated
by the **human auditor** (the owner). This page documents the tooling; it
makes zero model calls by construction.

## Scope — which references are audited

The benchmark builder (`scripts/m2_bench_build.py`) assigns every reference
a **provenance tier**: `owner` = a human transcription (HTR-pilot owner
annotations of handwritten lines; the owner-verified exam-002 cells) or
`text-layer` = the born-digital PDF's embedded text (printed RTL blocks,
printed mixed He/En, printed formulas, option rows paired from word
geometry). Only the former can be wrong in ways a human audit can fix.

**Rule** (`refaudit.is_manual_reference`): eligible iff the item's tier is a
human-transcription tier AND its reference provenance is not a
mechanical/text-layer derivation. Category is never the criterion — a
handwritten owner-transcribed mixed/formula line would be in scope; a
printed text-layer item of any category is not.

Result on the frozen benchmark: **102 eligible manual-reference items**
(86 `handwritten_line` + 16 `handwritten_cell`, writers e002–e007) and 27
out-of-scope text-layer items that keep their original references
unchanged. All denominators (progress, navigation, freeze, manifest,
preview, verifier prep) use the 102; the 129-item benchmark itself is
never altered.

## Launch

```powershell
.\.venv\Scripts\python.exe -m streamlit run scripts\reference_audit_ui.py -- --browser.gatherUsageStats false
```

(the flag keeps Streamlit's usage telemetry off — the repo's no-network
convention)

Per item: crop image and reference side by side, editable audited
transcription (RTL), optional note, and three decisions — **CONFIRM**
(original is exactly right; refused if the text was edited), **CORRECT**
(saves the edited text; both versions preserved), **AMBIGUOUS** (undecidable;
entered text preserved if any). Navigation: Prev/Next, filter by status,
jump-to-item, first-unchecked; the sidebar shows
confirmed/corrected/ambiguous/remaining progress. Every decision is saved
**atomically the moment the button is pressed** — closing or rerunning the
app never loses work, and reopening resumes at the first unchecked item.
Reset is per-item; the global reset requires typing `RESET`.

## Files

| File | Role |
|---|---|
| `evaluation/hebrew_bench_v2/reference_audit.json` | audit state (separate; originals never touched) |
| `evaluation/hebrew_bench_v2/reference_audit_manifest.json` | frozen manifest (content hash; only when 0 unchecked) |
| `evaluation/hebrew_bench_v2/audit_metrics_preview.json` | historical CER preview (read-only over persisted outputs) |
| `evaluation/hebrew_bench_v2/verifier_bench/` | verifier benchmark cases (emitted only after freeze) |

`references.json`, `items.json`, `outputs/**`, and `m2_bench_results.csv`
are read-only to this tooling — verified by tests
(`tests/test_reference_audit.py::test_original_reference_files_are_never_modified`).

## Scoring resolution (`scripts/refaudit.py::reference_for_scoring`)

| Status | Reference used | Strict CER |
|---|---|---|
| confirmed | audited (= original) | yes |
| corrected | audited | yes |
| ambiguous | flagged — never silently ordinary ground truth | no (exclude or report separately) |
| unchecked | `mode="final"` **refuses** (`UncheckedReferenceError`); `mode="preview"` returns the original marked not-audited | no |

## CLI

```powershell
.\.venv\Scripts\python.exe scripts\refaudit.py summary
.\.venv\Scripts\python.exe scripts\refaudit.py preview
.\.venv\Scripts\python.exe scripts\refaudit.py freeze
.\.venv\Scripts\python.exe scripts\refaudit.py verifier-prep [--emit]
```

`preview` reports, per historical config with persisted outputs: old CER
mean vs audited CER mean (same compared set), affected items, corrections,
ambiguous excluded/reported, unchecked not compared. It writes only
`audit_metrics_preview.json`; historical results keep their provenance and
are never rewritten. `freeze` refuses while any item is unchecked.
`verifier-prep` always prints dry counts (correct candidates vs real error
candidates, error kinds: omission / substitution / unsupported_addition /
number_sign_formula, harvested from audited references × persisted
outputs); `--emit` refuses until the audit manifest is frozen and current,
then writes **two** files: `cases_inputs.jsonl` (the ONLY model-visible
content — case id, item id, crop path, candidate transcription) and
`cases_labels.jsonl` (evaluation-side verdicts/error kinds/references that
must never enter a verifier prompt). Synthetic corruptions are deliberately
NOT generated yet — that happens after the freeze, as a separate step.

## Production verifier crops (Part 8 verdict)

Benchmark crops (A) exist — the 129 audited PNGs — and are the B2
substrate. Production explanation-crop geometry (B) **does not exist**:
`PageRegion` locations are descriptive text by design ("bottom third" —
`schema.py:157-160`), templates carry no coordinate fields, tablecrop's
calibrated geometry covers MC answer-table rows only, and lazy explanation
OCR sends whole labeled pages. The missing producer is a calibrated
per-(question, sub-item) explanation-region cropper; until one is built and
validated, production `ocr_verify` wiring stays unresolved — coordinates
are not to be invented, and full-page fallback would need an explicit
reviewed design.

## After the audit completes

1. `refaudit.py freeze` → manifest with content hash. *(done 2026-08-22)*
2. `refaudit.py preview` → attach to the audit record. *(done)*
3. `refaudit.py verifier-prep --emit` → the RAW B2 pool
   (`verifier_bench/`, 690 cases, kept byte-identical). *(done)*
4. `scripts/verifier_select.py propose` → composition report +
   `verifier_bench/selection_proposal.json` (positives + deduplicated,
   coverage-selected negatives, writer-level split proposals A/B); owner
   reviews the split; then `verifier_select.py --split <A|B> freeze` →
   `verifier_bench/selected/`. Then, separately, design synthetic
   near-miss corruptions against the frozen references (the real error
   pool is mostly severe; subtle errors are scarce).
5. Only then: model benchmarking per docs/model-selection.md (primary
   metric for OCR_VERIFY: false accept rate).
