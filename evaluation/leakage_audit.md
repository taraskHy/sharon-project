# Leakage audit — 2026-07-13

Question: can the system "cheat" by reading grade-bearing signals instead of
grading independently? Channels audited: filename grades, visible instructor
totals, red per-question scores/deductions/comments.

## Controlled probe (live model)

`autograder audit-leakage --split train --limit 5`: for each exam the model
is directly ASKED to read any instructor grade/scores it can see, once on
the original pages and once on masked pages
(`qwen3-vl:8b-instruct`, 1000 px, 5 probe pages/exam):

| Exam | Unmasked: grade extracted? | Masked: grade extracted? |
|---|---|---|
| exam-005 | no (sees score-like marks, cannot read a grade) | no |
| exam-006 | no | no |
| exam-011 | no | no |
| exam-012 | no | no |
| exam-013 | no | no |

**Result: 0/10 probes leaked a grade** (`eval_out/leakage_audit.json`).
Caveat, stated honestly: the prober is the same 8B model whose
handwriting-reading is weak — a stronger model might extract totals where
this one cannot. The audit should be repeated in the 32B bake-off, and
masking remains ON for all batch grading regardless.

## Structural guarantees (unit-tested, network-blocked tests)

- **Filename grades**: the model-visible payload never contains the source
  filename, path, or grade token — a test records every request and asserts
  their absence; batch runs use anonymized ids (`exam-00N`) everywhere, and
  the single-exam rerun driver grades an anonymized COPY path.
- **Expected grades** are read from the manifest only AFTER the prediction
  is finalized and saved (code path in `evalcli.py`, explicit comment), and
  never enter any prompt.
- **Instructor red ink** is masked from the rendered pages before inference
  (auditable per-page regions, originals untouched — masking tests), the
  close-read re-render goes through the same masked loader, prompts exclude
  grader ink, score fractions are deterministically dropped from
  convention notes, and the authority pass demotes any "answer" whose only
  evidence is grader annotation.
- **Grading direction**: Stage A errors were uniformly NEGATIVE
  (under-scoring, −26…−56) — the opposite signature of grade copying: a
  system reading the instructor's totals would trend toward them.

## In-grading observation

On exam-003 Q1.3 the student wrote D and the instructor overwrote a red E:
the pipeline extracted **D** in both audited runs — the instructor's
correction did not become the student answer (owner-verified ground truth).

## Verdict

No evidence of grade leakage on the probed sample; all designed barriers
are active and unit-tested. Standing risks: stronger-model re-audit
pending; annotation separation depends partly on masking (color-based) —
non-red instructor ink would rely on the prompt/authority layers, which is
why annotation-separation uncertainty is review-flagged when detected.
