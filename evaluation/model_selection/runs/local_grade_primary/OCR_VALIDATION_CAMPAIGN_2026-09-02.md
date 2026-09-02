# OCR validation campaign — FROZEN, NOT EXECUTED (2026-09-02 03:12:54)

`OCR_VALIDATION_CAMPAIGN_2026-09-02.json` sha `8ce4f5eea7b233c5…`; git `30c2a69521c9`.

- stage 1: 8 frozen bench smoke cases (m2-strict-v1, historical comparability)
- stage 2: 46 seen cases / 54 crops (production ocr-v1 + ExplanationTranscription), blocked on two named prep items (seen46-ocr subset registration; per-writer WER scoring)
- candidates: google/gemini-3.7-flash, openai/gpt-5.6-luna, anthropic/claude-sonnet-5 (primary google/gemini-3.7-flash; all UNSELECTED)
- budget: <= 62 calls/candidate, hard bound $2.00 campaign-total inside the global $10 ceiling
- request content: crop + exact-transcription instruction + minimal hint/schema ONLY; rubric/solution/grades/RAG are boundary-refused
- OCR calls executed by this freeze: **0**; OpenRouter authentication: **none**
