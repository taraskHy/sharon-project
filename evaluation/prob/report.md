# Prob-dataset evaluation (multiple-choice benchmark)

- Generated: 2026-08-07T06:24:14
- Job: `prob-eval-2026-08-07` — 13/13 exams graded
- Backend: `qwen3-vl:8b-instruct` at `http://localhost:11434/v1`
- Machine: Windows-11-10.0.26200-SP0 — **CPU-only, no GPU/VRAM**

## Total-score metrics vs official grades (grades.csv)

| Metric | Value |
|---|---:|
| Exams scored | 13 |
| Failures | 0 |
| Exact-grade accuracy | 46% |
| Within ±5 | 46% |
| Within ±10 | 85% |
| MAE | 6.923 |
| Median abs error | 10.0 |
| RMSE | 10.0 |
| Mean signed error | -5.385 |
| Max abs error | 20.0 |
| Human-review rate | 62% |
| Mean runtime / exam | 404.3 s |

## Total-score metrics vs audited sheets

The independent visual audit (manual_audit.json: every row double-
read unanimously, key re-derived from the booklets) found grades.csv
inconsistent with the physically marked sheets on scans 05 (+10),
06 (+10) and 13 (−10) — instructor totaling errors. Against the
sheet-faithful reference:

| Metric | Value |
|---|---:|
| Exams scored | 13 |
| Failures | 0 |
| Exact-grade accuracy | 69% |
| Within ±5 | 69% |
| Within ±10 | 85% |
| MAE | 4.615 |
| Median abs error | 0.0 |
| RMSE | 8.771 |
| Mean signed error | -4.615 |
| Max abs error | 20.0 |
| Human-review rate | 62% |
| Mean runtime / exam | 404.3 s |

## Answer-extraction accuracy vs audited sheets

- Mean per-row accuracy over 13 audited exams: **92.3%** (10 rows each; '—' entries below are rows DEFERRED to human review, not misreads)
- **Auto-decided rows: 120/120 correct** vs the audit — zero silent errors.
- Deferred rows: 10, of which 10 carry the audited answer among their listed candidates.
- If the reviewer resolves each deferred row as the audit read it, totals match the audited sheets on **13/13** exams and the official grades on 10/13 (the remaining gap is exactly the three documented instructor totaling errors).
- A correct total with per-row errors would be visible here — totals are never accepted on cancellation.

## Per-exam results

| Exam | Src | Variant | Answers (1-10) | Predicted | Official | Audited | Err(official) | Row acc | Mismatches | Review | Runtime (s) |
|---|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|
| exam-001 | 02 | heart | `1:B 2:A 3:D 4:C 5:B 6:D 7:— 8:D 9:D 10:C` | 50.0 | 60 | 60 | -10 | 90% | 7:—≠D | 1 | 536.1 |
| exam-002 | 05 | heart | `1:A 2:A 3:A 4:D 5:C 6:D 7:A 8:D 9:C 10:D` | 40.0 | 50 | 40 | -10 | 100% | — | 0 | 290.4 |
| exam-003 | 06 | heart | `1:A 2:A 3:D 4:D 5:B 6:B 7:A 8:D 9:B 10:B` | 60.0 | 70 | 60 | -10 | 100% | — | 0 | 263.3 |
| exam-004 | 13 | spade | `1:C 2:A 3:B 4:A 5:A 6:C 7:D 8:B 9:C 10:B` | 40.0 | 30 | 40 | +10 | 100% | — | 0 | 277.0 |
| exam-005 | 15 | diamond | `1:D 2:C 3:C 4:D 5:B 6:A 7:D 8:B 9:B 10:A` | 70.0 | 70 | 70 | +0 | 100% | — | 0 | 264.4 |
| exam-006 | 21 | diamond | `1:D 2:C 3:C 4:C 5:B 6:C 7:D 8:B 9:A 10:B` | 80.0 | 80 | 80 | +0 | 100% | — | 1 | 30.2 |
| exam-007 | 24 | diamond | `1:C 2:C 3:C 4:— 5:D 6:C 7:B 8:D 9:B 10:A` | 30.0 | 30 | 30 | +0 | 90% | 4:—≠C | 1 | 52.7 |
| exam-008 | 28 | spade | `1:B 2:A 3:A 4:A 5:B 6:A 7:— 8:C 9:C 10:—` | 40.0 | 50 | 50 | -10 | 80% | 7:—≠D 10:—≠D | 2 | 743.5 |
| exam-009 | 29 | club | `1:A 2:A 3:— 4:D 5:B 6:C 7:A 8:— 9:B 10:B` | 30.0 | 50 | 50 | -20 | 80% | 3:—≠B 8:—≠D | 2 | 734.7 |
| exam-010 | 30 | heart | `1:C 2:A 3:D 4:B 5:D 6:D 7:— 8:C 9:C 10:D` | 20.0 | 20 | 20 | +0 | 90% | 7:—≠A | 1 | 527.7 |
| exam-011 | 32 | club | `1:C 2:A 3:A 4:— 5:D 6:A 7:B 8:C 9:C 10:A` | 40.0 | 40 | 40 | +0 | 90% | 4:—≠B | 1 | 547.8 |
| exam-012 | 36 | club | `1:C 2:A 3:B 4:B 5:D 6:D 7:B 8:D 9:C 10:A` | 50.0 | 50 | 50 | +0 | 100% | — | 0 | 297.0 |
| exam-013 | 37 | club | `1:A 2:D 3:C 4:B 5:C 6:A 7:C 8:— 9:— 10:C` | 10.0 | 30 | 30 | -20 | 80% | 8:—≠D 9:—≠B | 2 | 690.7 |