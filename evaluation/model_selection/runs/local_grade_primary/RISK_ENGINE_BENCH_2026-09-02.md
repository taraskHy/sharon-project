# Risk-engine performance bench (2026-09-02 03:04:28)

Engine `risk-engine-v1`, policy `prospective_noninvalid_v1`, synthetic structurally-replicated inputs; zero inference.

| cases | mean decide | p95 | throughput | peak mem | event size | log bytes | admin agg |
|---|---|---|---|---|---|---|---|
| 46 | 735.73µs | 934.8µs | 1358.2/s | 0.046MB | 1477.0B | 67941 | 0.0092s |
| 100 | 702.2µs | 862.5µs | 1422.9/s | 0.074MB | 1476.2B | 147618 | 0.0093s |
| 1000 | 699.22µs | 836.5µs | 1428.9/s | 0.552MB | 1477.0B | 1476959 | 0.0323s |
| 10000 | 717.66µs | 902.3µs | 1392.1/s | 3.811MB | 1477.0B | 14769603 | 0.2117s |

Projected risk-layer overhead per 100 explanation cases: **0.1017s**. Full-exam figures cover the risk layer ONLY — OCR and model inference latency are not measured here and no full-exam automation claim is made.
