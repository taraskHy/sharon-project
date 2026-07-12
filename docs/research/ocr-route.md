# OCR + LLM Architecture Route for Hebrew Exam Grading — Research Findings (verified 2026-07-12)

All claims below were verified live on 2026-07-12 via WebSearch, WebFetch, the GitHub API, and the HuggingFace API. URLs are given per claim.

## Bottom line

**The OCR+LLM route is NOT viable as the primary architecture for this task.** No open OCR component available in July 2026 can transcribe messy modern Hebrew cursive exam handwriting, and the OCR route inherently does not address mark detection (circles/X/bubbles), ink-colour separation, or document-level marking conventions — all of which would still require a vision model or brittle custom CV. A pure-VLM architecture is the correct route; classic OCR (Tesseract) is at most a cheap auxiliary for the printed-Hebrew layer. Notably, even the OCR field itself has moved to VLMs: Surya 2's OCR is now itself a vision-language model served via vllm/llama.cpp, and the only active Hebrew-HTR work on HuggingFace consists of Qwen3-VL / GLM-4.6V fine-tunes.

## Component-by-component findings

### 1. Tesseract 5.x (Hebrew)
- **License:** Apache-2.0 — clean for university use (https://github.com/tesseract-ocr/tesseract/blob/main/LICENSE, accessed 2026-07-12).
- **Hebrew:** `heb` traineddata exists; RTL supported since v3 (https://tesseract-ocr.github.io/tessdoc/). Third-party 2026 benchmarking (MF Smart Research practitioner guide, https://mf-sr.com/en/blog/ocr-hebrew-2026-practitioner-guide.html, seen via search results 2026-07-12; direct fetch returned 403) reports ~92–96% accuracy on clean modern printed Hebrew.
- **Handwriting:** officially incapable. The Tesseract FAQ states handwriting recognition "won't work very well, as Tesseract is designed for printed text" (https://tesseract-ocr.github.io/tessdoc/FAQ.html, accessed 2026-07-12).
- **Fit:** usable only for the printed exam-question layer, CPU-only friendly. Zero help for student answers, marks, or ink colours.

### 2. Kraken / eScriptorium + Hebrew HTR models
- **Licenses:** Kraken is Apache-2.0 (GitHub API `repos/mittagessen/kraken/license`, accessed 2026-07-12); eScriptorium is MIT (https://en.wikipedia.org/wiki/EScriptorium; repo https://gitlab.com/scripta/escriptorium).
- **Available Hebrew models:** all are for MEDIEVAL manuscripts in square/book scripts — BiblIA "general model for Medieval Hebrew manuscripts in square script," >97% char accuracy on its own validation set, dataset CC-BY-NC-SA-4.0 on Zenodo (https://dl.acm.org/doi/10.1145/3476887.3476896, https://zenodo.org/records/5167263); Sofer Mahir covers Tannaitic rabbinic manuscripts (https://github.com/dstoekl/sofer_mahir); Tikkoun Sofrim combines HTR with crowdsourced correction for medieval Midrash manuscripts (https://elijahlab.haifa.ac.il/tikkoun-sofrim/?lang=en, https://dev.clariah.nl/files/dh2019/boa/0568.html). The Kraken model zoo (https://kraken.re/6.0.0/advanced/repo.html) contains a Medieval Hebrew model but **no modern Hebrew cursive model** (searched 2026-07-12).
- **Critical mismatch:** Israeli students write modern cursive Hebrew, whose letterforms differ radically from square script — the HHD paper itself notes cursive Hebrew letters "are more circular and considerably vary from their equivalent Hebrew block letters" (https://tc11.cvc.uab.es/datasets/HHD_v0_1). Medieval square-script models are useless on exam handwriting. Training your own Kraken model would require thousands of transcribed lines of real student handwriting — a data-collection project, not an integration.
- **Note:** CC-BY-NC-SA on BiblIA's dataset/models would also raise NC questions for institutional grading even if the script matched.

### 3. Surya OCR (Datalab)
- **Licenses (exact):** code Apache-2.0; **weights under a modified AI Pubs OpenRAIL-M** (https://github.com/datalab-to/surya, MODEL_LICENSE fetched raw 2026-07-12). Attachment A, Commercial restriction: prohibited "for any purpose if You (your employer, or the entity you are affiliated with) generated more than … ($5,000,000) in gross revenue in the prior year, except where Your Use is limited to personal use or research purposes" (same for >$5M funding). A university exceeds $5M revenue, so **use must qualify as 'research purposes' — operational grading of real exams is a legal gray zone**, likely requiring a Datalab commercial license (https://www.datalab.to/pricing). The license also imposes attribution and a share-alike clause that extends the license **to the Output** (Section 8), plus remote-restriction rights (Section 9) — unattractive for institutional deployment.
- **Hebrew:** yes — the Surya 2 internal 91-language benchmark lists `he` Hebrew at 90.9% pass rate (printed text) (https://github.com/datalab-to/surya/blob/master/static/docs/multilingual.md, raw fetch 2026-07-12); Arabic 72.7%.
- **Handwriting:** README demos "Handwritten Notes" (English); no Hebrew-handwriting evidence anywhere.
- **Architecture note:** Surya 2's layout/OCR/table-rec "all share a single vision-language model" served by vllm (GPU) or llama.cpp (CPU/Apple Silicon) (README, raw fetch 2026-07-12) — i.e., the best "OCR" option is already a VLM, undercutting the premise of an OCR-vs-VLM split.

### 4. PaddleOCR 3.x / PaddleOCR-VL
- **License:** Apache-2.0 (code and PaddleOCR-VL weights; https://github.com/PaddlePaddle/PaddleOCR/blob/main/LICENSE, https://huggingface.co/PaddlePaddle/PaddleOCR-VL).
- **Hebrew: NOT SUPPORTED, period.** The classic multilingual table lists Arabic/Persian/Urdu but no Hebrew (http://www.paddleocr.ai/v2.9/en/ppocr/blog/multi_languages.html, accessed 2026-07-12). The PaddleOCR-VL technical report's Appendix B (109 languages) lists the Arabic-script group "Arabic, Persian, Uyghur, Urdu, Pashto, Kurdish, Sindhi, Balochi" — **no Hebrew** (https://ar5iv.labs.arxiv.org/html/2510.14528, accessed 2026-07-12; also confirmed via https://huggingface.co/PaddlePaddle/PaddleOCR-VL/discussions/12). Hebrew support remains an open community request (https://github.com/PaddlePaddle/PaddleOCR/discussions/12734). **Eliminated.**

### 5. docTR and EasyOCR
- **docTR:** Apache-2.0 (https://github.com/mindee/doctr), but pretrained recognition models cover Latin scripts (plus some CJK/Cyrillic community models); no Hebrew vocab/model — you'd have to train your own (https://github.com/mindee/doctr/discussions/837). **Eliminated.**
- **EasyOCR:** Apache-2.0, 80+ languages, but **Hebrew is not supported**; requests open since 2020, still unresolved as of the 2024-updated issue (https://github.com/JaidedAI/EasyOCR/issues/363, https://github.com/JaidedAI/EasyOCR/issues/1334, https://www.jaided.ai/easyocr/). **Eliminated.**

### 6. Dedicated open Hebrew-handwriting models (HuggingFace, searched via HF API 2026-07-12)
- Searches for "hebrew handwritten", "hebrew ocr", "hebrew htr", "trocr hebrew" return only **experimental, low-download, undocumented artifacts**: the `cyttic/*` TrOCR-Hebrew experiment series (top item `exp21-trocr-hebrew-directfit-frozen`, 990 downloads, **no declared license**, no eval numbers), `oln-1/hebrew-htr-trocr` (24 downloads, no license), `sivan22/testing-trOCR-hebrew-handwritten` (69 downloads). The author of the sivan22 TrOCR-Hebrew effort states outright that it did not succeed for lack of data — "30K lines, while Microsoft used … over 600M" (https://huggingface.co/spaces/sivan22/TrOCR-handwritten-hebrew/discussions/1).
- The `kohelet-splendour/*` series (qwen3-vl-hebrew-htr, glm-4.6v-hebrew-htr variants) shows the community now fine-tunes **VLMs** for Hebrew HTR — further evidence the OCR-model route is dead for this script.
- **Datasets:** HHD (Ben-Gurion U., ICFHR 2020) is isolated characters + words from ~1,000 hand-filled forms — a research benchmark, not a production line-level HTR training corpus (https://tc11.cvc.uab.es/datasets/HHD_v0_1, https://zenodo.org/records/4287442).
- **Transkribus** has 11 Hebrew/Yiddish models (mostly historical), but most run only on its proprietary engine inside a paid SaaS (https://www.transkribus.org/blog/transcribe-handwritten-hebrew-yiddish-documents-ai-models) — fails the open/self-host requirement.

## Honest capability assessment

1. **Messy modern Hebrew exam handwriting: no open OCR component can transcribe it today.** Tesseract says so itself; Kraken has only medieval square-script models; Surya has no Hebrew-handwriting evidence; PaddleOCR/EasyOCR/docTR have no Hebrew at all; HF TrOCR fine-tunes are failed/experimental for lack of training data. This is a data-scarcity problem (no large modern-Hebrew-cursive line corpus exists), not a tooling gap.
2. **Mark detection and ink separation are outside OCR's scope entirely.** All of these tools output text lines + bounding boxes. Circles, X marks, filled bubbles, cross-outs, and overwrites on printed tables are not text; blue-vs-red ink separation is a colour-segmentation problem (doable with OpenCV HSV thresholding, but brittle against ballpoint-blue vs stamp-red variation, scanner colour shift, and overlapping strokes); document-level convention inference ("X means selected on this paper") is a reasoning task. The OCR route would therefore still need a vision model (or a fragile custom CV pipeline) for exactly the sub-tasks that decide grades.
3. **The LLM half is not the bottleneck** — open LLMs judge Hebrew text fine — but an LLM can only judge what reaches it; with handwriting transcription unsolved by OCR, the pipeline's input is garbage for the answers that matter.

## Verdict

OCR+LLM: **not viable** for this task in July 2026. Recommended role for OCR components: optional Tesseract (Apache-2.0, CPU) pass over the printed question text to give the VLM anchor text / cheap page indexing. Everything else — handwriting, marks, ink colours, conventions, semantic judging with structured JSON — needs the pure-VLM route (evaluated in a separate subtask). If a component-level fallback is ever wanted, Kraken (Apache-2.0) + a self-collected modern-Hebrew-cursive training set is the only license-clean path, and it is a multi-month data-labeling project. Surya is technically the strongest open OCR for printed Hebrew (90.9% on its internal benchmark) but its OpenRAIL-M weights license restricts >$5M organizations to "research purposes" and attaches share-alike terms to outputs — a poor fit for institutional exam grading without a Datalab commercial license.

## Key sources (all accessed 2026-07-12)
- Tesseract: https://github.com/tesseract-ocr/tesseract/blob/main/LICENSE ; https://tesseract-ocr.github.io/tessdoc/FAQ.html ; https://mf-sr.com/en/blog/ocr-hebrew-2026-practitioner-guide.html
- Kraken/eScriptorium: https://github.com/mittagessen/kraken (Apache-2.0 via GitHub API) ; https://kraken.re/6.0.0/advanced/repo.html ; https://gitlab.com/scripta/escriptorium
- Hebrew HTR projects: https://github.com/dstoekl/sofer_mahir ; https://elijahlab.haifa.ac.il/tikkoun-sofrim/?lang=en ; https://dl.acm.org/doi/10.1145/3476887.3476896 (BiblIA) ; https://zenodo.org/records/5167263
- Surya: https://github.com/datalab-to/surya (README, MODEL_LICENSE, static/docs/multilingual.md raw fetches)
- PaddleOCR: http://www.paddleocr.ai/v2.9/en/ppocr/blog/multi_languages.html ; https://ar5iv.labs.arxiv.org/html/2510.14528 (Appendix B) ; https://huggingface.co/PaddlePaddle/PaddleOCR-VL ; https://huggingface.co/PaddlePaddle/PaddleOCR-VL/discussions/12
- EasyOCR: https://github.com/JaidedAI/EasyOCR/issues/363 ; https://github.com/JaidedAI/EasyOCR/issues/1334 ; https://www.jaided.ai/easyocr/
- docTR: https://github.com/mindee/doctr ; https://github.com/mindee/doctr/discussions/837
- Hebrew handwriting models/datasets: HuggingFace API model searches (hebrew handwritten / hebrew htr / trocr hebrew) ; https://huggingface.co/spaces/sivan22/TrOCR-handwritten-hebrew/discussions/1 ; https://tc11.cvc.uab.es/datasets/HHD_v0_1 ; https://zenodo.org/records/4287442
- Transkribus (non-open comparison): https://www.transkribus.org/blog/transcribe-handwritten-hebrew-yiddish-documents-ai-models