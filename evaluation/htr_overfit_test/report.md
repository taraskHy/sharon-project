# Real-data overfit test — CRNN+CTC on 20 verified training lines

Date: 2026-07-16 21:08:23. Pre-registered gate: mean normalized line CER <= 0.05 AND >= 18/20 lines with CER <= 0.1.

## VERDICT: **PASS** — mean CER 0.0028 (raw 0.0028), 20/20 lines within 0.1.

## Setup
- 20 lines picked deterministically from 56 eligible (selected_ids.json written before training).
- No augmentation; train == val == the 20 lines (memorization rig).
- prepare: {"total_samples": 20, "kept_lines": 20, "excluded": {}, "written_images": 20, "skipped_unknown_char_lines": 0}
- epochs run: 900; trainer best (in-memory) CER 0.0028.

## Loss / memorization-CER by epoch (every 25th + last)
```
epoch   1 loss 19.9815 val_cer 1.0000  <- saved
epoch   2 loss 15.7332 val_cer 1.0000
epoch   3 loss 8.9765 val_cer 1.0000
epoch  26 loss 3.3734 val_cer 1.0000
epoch  51 loss 3.2656 val_cer 1.0000
epoch  76 loss 3.1409 val_cer 1.0000
epoch 101 loss 2.8332 val_cer 1.0000
epoch 126 loss 2.3616 val_cer 0.9916  <- saved
epoch 151 loss 1.7871 val_cer 0.9621  <- saved
epoch 176 loss 1.3330 val_cer 0.8631
epoch 201 loss 0.9856 val_cer 0.5994
epoch 226 loss 0.6784 val_cer 0.3461  <- saved
epoch 251 loss 0.4685 val_cer 0.2132  <- saved
epoch 276 loss 0.3149 val_cer 0.1625  <- saved
epoch 301 loss 0.2390 val_cer 0.1283
epoch 326 loss 0.1845 val_cer 0.0960
epoch 351 loss 0.1340 val_cer 0.0670  <- saved
epoch 376 loss 0.1100 val_cer 0.0472  <- saved
epoch 401 loss 0.0776 val_cer 0.0322  <- saved
epoch 426 loss 0.0678 val_cer 0.0207  <- saved
epoch 451 loss 0.0507 val_cer 0.0192
epoch 476 loss 0.0454 val_cer 0.0178
epoch 501 loss 0.0350 val_cer 0.0132  <- saved
epoch 526 loss 0.0321 val_cer 0.0117
epoch 551 loss 0.0251 val_cer 0.0100  <- saved
epoch 576 loss 0.0221 val_cer 0.0088
epoch 601 loss 0.0203 val_cer 0.0074
epoch 626 loss 0.0198 val_cer 0.0057
epoch 651 loss 0.0183 val_cer 0.0057
epoch 676 loss 0.0148 val_cer 0.0057
epoch 701 loss 0.0124 val_cer 0.0057
epoch 726 loss 0.0119 val_cer 0.0057
epoch 751 loss 0.0101 val_cer 0.0057
epoch 776 loss 0.0106 val_cer 0.0042
epoch 801 loss 0.0091 val_cer 0.0042
epoch 826 loss 0.0084 val_cer 0.0042
epoch 851 loss 0.0083 val_cer 0.0042
epoch 876 loss 0.0079 val_cer 0.0042
epoch 898 loss 0.0068 val_cer 0.0028
epoch 899 loss 0.0069 val_cer 0.0028
epoch 900 loss 0.0066 val_cer 0.0028
```

## Aggregates
- final training CER: normalized 0.0028, raw 0.0028; WER 0.0187.
- RTL order: mean CER of REVERSED predictions 0.847 vs 0.0028 correct-order — reversed must be far worse.
- mean confidence 0.9557 (min 0.9416).
- checkpoint save/reload: trainer-best 0.0028 vs reloaded-decode 0.0028 (|delta| 0.0000).
- vocabulary: 71 distinct label chars; missing from syms: none; lines skipped for unknown chars: 0.
- image widths after h=128 resize: min 654, max 1916 px; CTC length violations (frames < symbols): none.
- empty decodes: none.

## Per-line: prediction vs ground truth
| line | CER | conf | text (GT then PRED) |
|---|---|---|---|
| e003_q1_r1__l1 | 0.000 | 0.964 | GT: ניתן לראות ברמות התדרים הגבוהים שיש שפות בכיוון התנועה(הטשטוש) |
| | | | PR: ניתן לראות ברמות התדרים הגבוהים שיש שפות בכיוון התנועה(הטשטוש) |
| e003_q1_r2__l2 | 0.000 | 0.942 | GT: רק התדרים הגבוהים נשמרו ב-High pass לכן התדרים הנמוכים מאופסים (כולל dc) |
| | | | PR: רק התדרים הגבוהים נשמרו ב-High pass לכן התדרים הנמוכים מאופסים (כולל dc) |
| e003_q1_r3__l1 | 0.000 | 0.944 | GT: עוצמת התדרים הגבוהים אחרי הפעולה גדלה ולכן הרמות הנמוכות מראות שפות עבות יותר |
| | | | PR: עוצמת התדרים הגבוהים אחרי הפעולה גדלה ולכן הרמות הנמוכות מראות שפות עבות יותר |
| e003_q1_r4__l1 | 0.000 | 0.950 | GT: ניתן לראות שרק תדרים נמוכים נשמרו ולכן כל הרמות של התדרים הגבוהים מאופסות |
| | | | PR: ניתן לראות שרק תדרים נמוכים נשמרו ולכן כל הרמות של התדרים הגבוהים מאופסות |
| e003_q1_r5__l1 | 0.000 | 0.958 | GT: הרעש והטשטוש שלו בכיוון ציר ה-x מובלט בתדרים הגבוהים |
| | | | PR: הרעש והטשטוש שלו בכיוון ציר ה-x מובלט בתדרים הגבוהים |
| e003_q1_r6__l1 | 0.000 | 0.959 | GT: הרעש והטשטוש שלו בכיוון ציר ה-y מובלט בתדרים הגבוהים |
| | | | PR: הרעש והטשטוש שלו בכיוון ציר ה-y מובלט בתדרים הגבוהים |
| e003_q1_r7__l1 | 0.000 | 0.976 | GT: הרמה העליונה שכוללת DC בהירה יותר |
| | | | PR: הרמה העליונה שכוללת DC בהירה יותר |
| e003_q1_r8__l1 | 0.000 | 0.957 | GT: ברמה העליונה שכוללת DC ניתן לראות שערכים בהירים יותר, נהיו יותר בהירים |
| | | | PR: ברמה העליונה שכוללת DC ניתן לראות שערכים בהירים יותר, נהיו יותר בהירים |
| e003_q2_r1__l1 | 0.000 | 0.961 | GT: כל פיקסלי השפות שנקלטו קיבלו  ערך 255 לעומת שאר הפיקסלים(הרוב) |
| | | | PR: כל פיקסלי השפות שנקלטו קיבלו  ערך 255 לעומת שאר הפיקסלים(הרוב) |
| e003_q2_r2__l1 | 0.000 | 0.948 | GT: טשטוש הוריד רעש ולכן פחות פיקסלים נקלטו בשפהב והערך 255 בהיסטוגרמה ירד |
| | | | PR: טשטוש הוריד רעש ולכן פחות פיקסלים נקלטו בשפהב והערך 255 בהיסטוגרמה ירד |
| e003_q2_r3__l1 | 0.000 | 0.967 | GT: מפסר הפיקסלים לא השתנה והערכים מתפרסים על כל ספקטרום הערכים האפשרי |
| | | | PR: מפסר הפיקסלים לא השתנה והערכים מתפרסים על כל ספקטרום הערכים האפשרי |
| e003_q2_r4__l1 | 0.021 | 0.960 | GT: ניתן לראות שלעומת Uniform יש פחות ערכים אפשריים |
| | | | PR: ניתן לראות שלעומת Uniform יש פחות ערכים אפשרים |
| e003_q2_r5__l1 | 0.000 | 0.942 | GT: לכל BIN(20 סה"כ) יש ערך שנבחר בצורה אחידה וניתן לראות שבהם חולקו הפיקסלים |
| | | | PR: לכל BIN(20 סה"כ) יש ערך שנבחר בצורה אחידה וניתן לראות שבהם חולקו הפיקסלים |
| e003_q2_r6__l1 | 0.000 | 0.944 | GT: סכום המסכה הוא גדול מ-1, לכן טווח הערכים גדול מהטווח בהסטוגרמה המקורית ויותר פרוס |
| | | | PR: סכום המסכה הוא גדול מ-1, לכן טווח הערכים גדול מהטווח בהסטוגרמה המקורית ויותר פרוס |
| e003_q2_r7__l1 | 0.000 | 0.948 | GT: הוכפל מספר הפיקסלים בכל התמונה פי 2 והוכפל מספר הפיקסלים בכל ערך |
| | | | PR: הוכפל מספר הפיקסלים בכל התמונה פי 2 והוכפל מספר הפיקסלים בכל ערך |
| e003_q2_r8__l1 | 0.000 | 0.952 | GT: ניתן לראות שמספר הפיקסלים בכל עוצמה הוזז ימינה באותו מרחק שהיה במקור. |
| | | | PR: ניתן לראות שמספר הפיקסלים בכל עוצמה הוזז ימינה באותו מרחק שהיה במקור. |
| e004_q1_r1__l1 | 0.034 | 0.958 | GT: הWAVEELETS מטושטשת בכל החלקים |
| | | | PR: הWAVEELETS מטושטשת בכל החלים |
| e004_q1_r2__l1 | 0.000 | 0.957 | GT: התדר הנמוך נעלם בכלל HighpassFilter |
| | | | PR: התדר הנמוך נעלם בכלל HighpassFilter |
| e004_q1_r3__l1 | 0.000 | 0.959 | GT: התמונה מחודדת יותר ושמרה על הצבעים |
| | | | PR: התמונה מחודדת יותר ושמרה על הצבעים |
| e004_q1_r4__l1 | 0.000 | 0.969 | GT: התדר הגדול(High) נעלם בגלל ה Low Pass Filter |
| | | | PR: התדר הגדול(High) נעלם בגלל ה Low Pass Filter |

Total wall time 2663s.
