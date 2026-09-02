# Paired 32-crop OCR experiment — artifact generators

These are the scripts that produced the committed artifacts under
`evaluation/model_selection/runs/ocr_primary/OCR_SEEN32_*`. They were written
during the run and are committed here because an independent verifier correctly
observed that the artifacts were **not regenerable from committed code**.

Every script is **read-only with respect to raw outputs** and makes **zero
inference calls**. They read `outputs.jsonl`, `run.json`, the frozen manifest and
the gateway ledger, and write derived artifacts.

`build_metrics.py` additionally makes one **metadata** request to OpenRouter's
`/key` endpoint to reconcile the local ledger against the provider's own usage
figure. It is skipped when `OPENROUTER_API_KEY` is absent, so the script runs
fully offline; the key is read from the environment and never printed, logged or
persisted.

| Script | Produces |
|---|---|
| `build_metrics.py` | `OCR_SEEN32_PAIRED_RESULT_2026-09-02.json` (case matrix, stratified metrics, reliability bounds, fallback replay, routing comparison, accounting) |
| `build_e2e_replay.py` | `OCR_SEEN32_E2E_REPLAY_2026-09-02.json` |
| `build_performance.py` | `OCR_PERFORMANCE_2026-09-02.json` |
| `build_readiness.py` | `OCR_SHIPMENT_READINESS_2026-09-02.json` |
| `build_next_experiment.py` | `experiments/OCR_PROMPT_V2_NEUTRAL_FRAMING_2026-09-02.json` |
| `build_report_md.py` | `OCR_SEEN32_PAIRED_RESULT_2026-09-02.md` |

## Honest limitation

Running `build_metrics.py` alone will **not** reproduce the committed JSON
byte-for-byte. The committed artifact carries corrections applied on top after
independent adversarial verification — the composite-refutation block, the
rescue-quality refutation, the request/cache accounting split and the
critical-error accounting note. Those corrections were applied as separate
passes and are recorded inside the artifact under
`verification_corrections`, `routing_comparison._composite_refutation` and
`sonnet_as_fallback.REFUTATION_after_verification`.

The *measurements* regenerate exactly; the *narrative corrections* are additive
and are documented in place rather than recomputed. Timestamps also differ on
every regeneration, which changes the content hash.

Run from the repo root with the project venv:

```
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe scripts/ocr_seen32/build_metrics.py
```
