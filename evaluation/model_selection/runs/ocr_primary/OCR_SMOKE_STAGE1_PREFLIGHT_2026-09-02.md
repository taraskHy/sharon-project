# OCR stage-1 smoke — PREFLIGHT ONLY, execution blocked (2026-09-02 11:42:20)

git `3bd7b1779f8e`; campaign `8ce4f5eea7b233c5…`; smoke selection `cc5c9f1ff9911a68…`; schema `003ee19969c8c443…`. **Provider requests made: 0. Additional spend: $0.**

## Verified (zero cost)

- 8 frozen audited DEV cases (5 handwritten e002/e003, 3 printed text-layer), each with crop sha256, reference sha256 and admissible provenance; HELD_OUT = 0
- request contract: one crop + frozen exact-transcription prompt, no text blocks, 0 grading tripwire hits; boundary refuses all six grading roles; 8 OCR prompts registered
- live catalog: all three slugs present; pricing drift — gemini live 2x the local table
- dry-runs (mode dry_run, 0 calls, identical 8-case population): gemini $0.009707 (live-adjusted $0.019414), luna-pro $0.005817, sonnet $0.051773

## Predicted-cost gate ($0.05)

- three arms: **0.067297 local / 0.077004 live-adjusted -> FAIL**
- gemini + luna-pro: 0.015524 / 0.025231 -> PASS; sonnet alone 0.051773 -> BLOCKED_BY_CEILING

## Blockers

- **A** — OPENROUTER_API_KEY absent (process, user and machine scope; no .env, no dotenv loader). Resolution: owner exports the key in their own shell before invoking the run (never pasted into chat or a file)
- **B** — the frozen campaign's recorded paid commands omit --research; the runner refuses a live remote ocr_primary bench without it. Resolution: append --research (the project-designed cloud-benchmark mode; the request contract was verified independently and the per-request leakage check is unchanged)
- **C** — predicted three-arm cost 0.067297 (local) / 0.077004 (live-adjusted) exceeds the $0.05 ceiling; claude-sonnet-5 alone predicts 0.051773. Resolution: either raise the operational ceiling to >= $0.08 for the three-arm smoke, or authorize gemini + luna-pro now (predicted 0.025 live-adjusted) and defer sonnet

## Exact commands once the owner exports the key in their own shell

- gemini: `python -m autograder bench run --role ocr_primary --split dev --subset smoke --candidate google/gemini-3.7-flash --backend openrouter --models-config models.toml --research --i-understand-this-spends-money`
- luna_pro: `python -m autograder bench run --role ocr_primary --split dev --subset smoke --candidate openai/gpt-5.6-luna-pro --backend openrouter --models-config models.toml --research --i-understand-this-spends-money`
- sonnet_requires_ceiling_authorization: `python -m autograder bench run --role ocr_primary --split dev --subset smoke --candidate anthropic/claude-sonnet-5 --backend openrouter --models-config models.toml --research --i-understand-this-spends-money`
