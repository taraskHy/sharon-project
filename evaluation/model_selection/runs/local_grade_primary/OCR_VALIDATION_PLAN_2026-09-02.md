# OCR validation plan — PREPARED, NOT EXECUTED (2026-09-02)

**Standing blocker, explicitly preserved: production OpenRouter OCR quality is
still unvalidated.** Every grading result in this campaign was computed over
FROZEN, human-audited transcriptions. A local-grader improvement (or its
absence) establishes nothing about end-to-end production quality, where the
transcription itself comes from an OCR role that has never been benchmarked
against the audited references.

## Plan (owner-gated; no step below was run)

1. **Population**: the same frozen SEEN-46 cases (DEV+CALIBRATION), whose
   audited transcriptions are the OCR ground truth (pixel-exact crop geometry
   already derived; evidence repairs already applied). HELD_OUT stays sealed.
2. **Pre-registration**: freeze an OCR_VALIDATION spec (case list, crop
   hashes, candidate OCR routes, CER/WER metric definitions, acceptance
   thresholds, git commit) before any call — same discipline as the grading
   experiments.
3. **Candidates**: the OCR_PRIMARY role candidates already registered in
   `candidates.toml` (cloud allowed for OCR under the production boundary:
   CLOUD_OCR_ALLOWLIST). Budget from the existing $10 evaluation budget;
   smoke subset first, then the full 46.
4. **Metrics**: per-line CER/WER against the audited transcriptions
   (writer-grouped, per the earlier writer-grouped-CER finding), plus a
   downstream metric: re-run the FROZEN local grader on OCR output vs frozen
   transcription and measure verdict flips (transcription sensitivity).
5. **Gates**: propose CER <= 5% per writer AND zero harmful verdict flips on
   the seen set before OCR feeds grading in production; calibrate on
   CALIBRATION only; report writer-held.
6. **Separation**: OCR validation NEVER changes grading prompts, references,
   or frozen outputs; grading improvements NEVER validate OCR.

Execution requires an explicit owner go-ahead (cloud spend + key use).
