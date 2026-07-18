# Claude Vision assisted-annotation benchmark — pre-registered protocol

Written 2026-07-17, BEFORE any request to the Anthropic API and before any
candidate generation. Owner-authorized exception to the project's
"local models only" rule, scoped to THIS experiment exactly as specified
in the owner's instruction of 2026-07-17.

## Privacy pre-flight (confirmed before anything else)

**This experiment sends anonymized handwriting crops to Anthropic.**
Exactly two kinds of image leave the machine, both from the
validator-checked pilot package (`evaluation/htr_pilot/images/`, writer
codes e0NN, no grade-bearing names — validator: RESULT PASS):

1. the cleaned, student-ink-only LINE crop (`*_lN.png`);
2. in pass B only, the cleaned full explanation-cell image
   (`*_cell_clean.png`) of the SAME student — cleaned means printed
   table structure and red instructor ink are removed by the build
   pipeline.

Payloads contain ONLY: base64 image bytes + the fixed instruction text
below. **Never sent:** file names or sample ids, verified
transcriptions, question text, answer keys, rubrics, course
vocabulary, neighboring students' answers, instructor scores, grades,
`cell_orig` images (they can carry instructor ink), or anything from
`htr_pilot_sources.json`. The generation script cannot read
annotations at all (enforced by test). Anthropic API data is not used
for training by default; the repository stays private.

## Claude output is never ground truth

Candidates are stored under `evaluation/claude_candidates/outputs/`
with `"verified": false`, never inside `annotations/`, and nothing in
this experiment writes or marks any annotation record. Only the owner,
in the annotation app, can create a verified transcription.

## Benchmark set (deterministic, fixed now)

30 owner-verified train lines: for each writer in (e004, e005, e006),
the first 10 lines in sorted sample_id order whose annotation is
status `ok`, human_verified, WITHOUT a partial [לא קריא] span
(nonblank + correctly segmented by definition of `ok`), and NOT among
the 20 overfit-test ids. e003 is excluded (16/16 of its clean lines
were overfit-training lines), e007 has no eligible line. Selection
recorded ids-only in `claude_bench_ids.json` before generation.

## Passes (both claude-opus-4-8, the current most capable Opus tier)

- **A `claude_line`** — line crop only.
- **B `claude_line_cell`** — cleaned cell (context) + line crop, plus
  the fixed sentence: "The first image is the student's full answer
  cell (context only). The second image is a single line cropped from
  that cell. Transcribe ONLY the line shown in the second image."

Request parameters (fixed): model `claude-opus-4-8`, system = the
owner's instruction verbatim (below), `max_tokens` 16000, adaptive
thinking (`{"type": "adaptive"}`), no sampling parameters (removed on
Opus 4.8). Prompt version tag: `claude_htr_v1`.

System instruction (verbatim from the owner):

> Transcribe exactly the handwritten text visible in the image.
> Preserve spelling mistakes, punctuation, English words, numbers, and formulas.
> Do not explain, correct, complete, or infer likely words.
> Use [לא קריא] for any unreadable span.
> Return only the transcription.

Saved per call BEFORE evaluation: full raw content JSON, response
model id, prompt version, image SHA256(s), input type, ISO timestamp,
latency, stop_reason, token usage, parsed candidate. GT is read only
by `scripts/claude_candidates_eval.py`, strictly after all raw
outputs exist.

## Metrics (campaign definitions, `scripts/hebrew_bench_eval.py`)

Normalized CER = char-Levenshtein / len(ref) after the campaign
normalization; WER, omission rate (word deletions / ref words),
insertion rate (insertions / hyp words). Two exact-match definitions,
both reported: **strict** = equality after NFC + bidi-control-strip +
whitespace collapse (`htr_annotation_lib.normalize_text`); **normalized**
= equality after campaign normalization (punctuation/niqqud-stripped).
Edit classes: no_edit = strict-exact; minor = CER ≤ 0.15; moderate ≤
0.40; major > 0.40. **Major hallucination** (fixed now): a line whose
word alignment contains ≥ 2 inserted words.

**Estimated annotation time saved** (fixed model, reported as an
estimate, not a measurement): typing from scratch = max(15, 0.55 ×
ref_chars) seconds; with a candidate shown: 6 s if strict-exact,
8 + 1.2 × char_edits s if CER ≤ 0.15, else scratch + 5 s wasted review.

## Acceptance gate for adding suggestions to the app (owner's, verbatim — not to be weakened after results)

- at least 40 % exact-match lines;
- median CER at most 0.10;
- no major hallucination on more than 5 % of lines;
- agreement or confidence must identify a subset whose CER is at most 0.10.

Operationalization fixed now: the gate is evaluated per pass (A and B);
"exact-match" uses the NORMALIZED definition (the more generous one);
the agreement clause passes iff some A↔B agreement threshold
τ ∈ {1.0, 0.98, 0.95, 0.9, 0.85, 0.8} selects ≥ 5 lines whose mean CER
≤ 0.10 (Claude Vision returns no usable confidence signal, so
agreement is the only model-visible selector). The experiment is
**ACCEPT** iff at least one pass meets all four clauses.

If REJECT: no candidates are generated for untouched lines, the
annotation interface is not modified except to record the negative
result, and the experiment stops.
