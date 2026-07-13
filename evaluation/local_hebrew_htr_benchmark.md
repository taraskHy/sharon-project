# Local Hebrew HTR benchmark (iterations 6–8 of the transcription campaign)

Constraint set: fully local (no Anthropic/hosted vision APIs), separate
virtual environment (`.venv-htr`: torch 2.13.0+cpu, transformers 5.13.1),
no arbitrary installers, safetensors preferred, `trust_remote_code=False`,
raw output only (no spell correction / LM repair / semantic completion /
answer-key context). Scored post-inference against the owner's 16 verified
exam-002 cells (11 strict, 5 hard), which never reach any recognizer.
Results ledger: `evaluation/local_hebrew_htr_results.csv`. Strongest-Qwen
reference: 8B + strict prompt + contrast = **CER 0.786, usable 0 %**.

## Candidate provenance cards (researched + adversarially verified, 2026-07-13)

### 1. HebHTR — best domain provenance, NOT runnable tonight (needs approval)
- Repo: github.com/Lotemn102/HebHTR, master@snapshot-6 checkpoint; repo
  ARCHIVED (read-only) 2024-03-03; **no license** (all-rights-reserved
  default → private research only).
- Trained for: modern Hebrew handwriting — README-verified: ~100 K words,
  "around 50,000 … real words, taken from students scanned exams", 25
  hands; other half synthetic from MILA *stopwords*. Self-reported
  **validation CER 4.76 %** (model/accuracy.txt).
- Input: word crops (128×32 binary); current master's page→word
  segmentation was REMOVED in Sep-2020 commits and the checked-in pipeline
  is broken as-is (color/gray shape bug; char-splitting extend bug) — usable
  only via commit fdc5ae0 or a custom wrapper.
- Stack: TensorFlow 1.12 + a precompiled Linux `TFWordBeamSearch.so`
  (opaque binary; known-CVE TF version) → **requires WSL2/Docker on this
  Windows box = system-level change → deferred pending owner approval**.
- Value tonight: **existence proof** — a small CTC model trained on real
  student-exam Hebrew reached ~4.8 % CER; the failure of general-purpose
  VLMs is not evidence that the task is hard for domain-trained HTR.

### 2. sivan22/hdd-words-ocr — iteration 6 candidate (RUN)
- HF `sivan22/hdd-words-ocr` @ `e089ce717594610492d8c53d9e35ec5b80b402bb`
  (2023-06-05). VisionEncoderDecoder (~0.24 B params), **model.safetensors
  verified present** and loaded with `use_safetensors=True`,
  `trust_remote_code=False`. **No license declared** → private research
  only. Trained for: modern Hebrew handwriting (HDD dataset lineage —
  author's `hebrew-handwritten-dataset`/`hebrew-words-dataset`); word-level
  input; no built-in segmentation.
- Pipeline evaluated: deterministic projection segmentation (line bands →
  RTL-ordered word bands; crops saved under
  `evaluation/hebrew_bench/segments_words/it6_hdd_words/`) → one greedy
  recognition per word → RTL join. CPU inference.
- Security: standard transformers classes only; safetensors; pinned
  revision; no remote code.

### 3. sivan22/ABBA-HTR — SKIPPED (pickle-only weights)
- HF `sivan22/ABBA-HTR`: line-level SwinV2 + BEREL decoder — but the repo
  ships **`pytorch_model.bin` only (pickle), no safetensors** (verified via
  the HF tree API). Per the security rules it is not loaded without
  explicit owner approval. Also: BEREL decoder suggests rabbinic-text
  orientation; no license declared.

### 4. Medieval-Hebrew kraken lineage — low fit, catalogued
- BiblIA (Zenodo 5468286, CC-BY-SA 4.0) and MiDRASH Geniza (Zenodo
  18732245, CC-BY-**NC**-SA) are real, licensed, line-level kraken Hebrew
  models — for **medieval** square/semi-cursive hands, a different script
  tradition from modern Israeli cursive. Catalogued as controls/fallbacks,
  not run tonight (kraken on native Windows is additionally fragile).
- CATMuS does NOT cover Hebrew (verified); Tikkoun Sofrim / Sofer Mahir are
  medieval/rabbinic and feed the BiblIA→MiDRASH lineage.

### 5. Multilingual document-OCR candidates (iteration 7 pool)
- **Chandra OCR 2** (datalab-to): modern-document handwriting focus (forms,
  homework); code Apache-2.0, **weights under an AI-Pubs OpenRAIL-style
  license** (terms to record before weight download); page-level.
- **Surya OCR 2** (datalab-to): best surveyed Hebrew score of dedicated OCR
  models (90.9 % — printed-leaning benchmark); tiny, CPU-runnable; same
  weights-license caveat.
- Ruled out with evidence: microsoft/trocr-*-handwritten (English-only,
  IAM); PaddleOCR classic + MMOCR + EasyOCR (no Hebrew recognition models —
  EasyOCR/Tesseract retained ONLY as printed-OCR baselines and their
  failure proves nothing about HTR); PaddleOCR-VL (no Hebrew in its
  technical report's language appendix + `trust_remote_code=True`);
  CHURRO-3B (peer-reviewed handwritten-Hebrew training but historical
  hands; Qwen research license).

## Fine-tuning path (required deliverable if no pretrained candidate is adequate)

Research-ranked for this project (modern cursive, 1–3-line table cells,
RTX 2000 Ada 15.4 GB):

1. **kraken (`ketos train`)** — native RTL/BiDi, line-level (matches the
   cells), tiny models, hours of training on this GPU; eScriptorium gives
   an operator correct-and-retrain loop. Data format: line images +
   PAGE/ALTO XML or plain pairs.
2. **PyLaia** (MIT, Teklia-maintained) — simplest data format (image+text
   table), fastest iteration; add a Hebrew n-gram LM rescoring later.
3. TrOCR-style only with a large synthetic-pretraining budget (decoder must
   be swapped for Hebrew → loses most pretraining value). DAN: wrong
   operating point. HTR-VT: niche from-scratch option (~6–8 K lines).

**Line-count estimate:** kraken documentation-class guidance ≈ 800 lines
for a usable single-hand model; multi-writer modern Hebrew needs more
(HebHTR calibration: ~50 K real words ≈ tens of writers → 4.8 % CER). The
41 development exams hold ≈ 16 explanation cells × ~1.5 lines ≈ **~1,000
verified lines total** — enough for a first multi-writer fine-tune wave
(with the owner as verifying annotator, reusing the annotation-package
workflow built today: crops → candidate → human_verified). Estimated
training: hours (kraken/PyLaia scale), not days, on the local GPU.

## Results (appended as iterations complete)
