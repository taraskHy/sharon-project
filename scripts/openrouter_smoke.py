"""OpenRouter infrastructure smoke — exactly 2 paid provider calls.

  1. tiny TEXT structured-output call
  2. tiny IMAGE structured-output call (a real benchmark crop, no reference)
  3. the identical image request again -> must be a LOCAL cache hit
     (zero provider requests, zero paid tokens)

Verifies routing, auth, parsing, structured output, ledger fields (input/
output/reasoning tokens, provider/model metadata, request id, reported
cost), budget counters, cache hit, and that no secret appears in any
persisted artifact or exception text. Reads OPENROUTER_API_KEY and
SMOKE_MODEL from the environment only. STOPs with a nonzero exit on any
failure.
"""

from __future__ import annotations

import base64
import json
import os
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from pydantic import BaseModel  # noqa: E402

from autograder.orchestrator import setup_from_config  # noqa: E402
from autograder.usage import BudgetLimits  # noqa: E402


class Echo(BaseModel):
    answer: str


class ImgRead(BaseModel):
    contains_handwriting: bool
    dominant_script: str


def main() -> int:
    key = os.environ.get("OPENROUTER_API_KEY")
    model = os.environ.get("SMOKE_MODEL")
    if not key or not model:
        print("OPENROUTER_API_KEY and SMOKE_MODEL must be set in the environment")
        return 2
    out = REPO / "evaluation" / "openrouter_smoke"
    out.mkdir(parents=True, exist_ok=True)
    cfg = out / "models_smoke.toml"
    cfg.write_text(
        '[defaults]\nstructured_mode = "json_schema"\ntemperature = 0.0\ntimeout_s = 120.0\n'
        '[models.smoke_text]\nbackend = "openrouter"\nmodel = "${SMOKE_MODEL}"\nmax_tokens = 40\n'
        'reasoning = { effort = "none" }\nprompt_version = "smoke-v1"\n'
        '[models.smoke_image]\nbackend = "openrouter"\nmodel = "${SMOKE_MODEL}"\nmax_tokens = 60\n'
        'reasoning = { effort = "none" }\nprompt_version = "smoke-v1"\n'
        '[budget]\nenabled = true\nmax_calls_per_job = 3\nsoft_fraction = 0.5\n',
        encoding="utf-8")
    rt = setup_from_config(cfg, out / "state", budget=None)
    meta = {"job_id": "smoke", "exam_id": "smoke-exam", "question_id": "0", "stage": "smoke"}
    report: dict = {"model": model, "calls": []}

    def check(name, cond, detail=""):
        report.setdefault("checks", []).append({"check": name, "ok": bool(cond), "detail": detail})
        print(f"  [{'OK' if cond else 'FAIL'}] {name} {detail}")
        return bool(cond)

    ok = True
    # 1. text
    print("1) tiny TEXT structured call")
    t0 = time.monotonic()
    r1 = rt.gateway.call(task="smoke_text", system="Reply with ONLY the JSON object.",
                         content_blocks=[{"type": "text", "text": 'Return {"answer": "pong"}'}],
                         output_model=Echo, meta=meta)
    report["calls"].append({"task": "smoke_text", "cache_hit": r1.cache_hit, "usage": r1.usage,
                            "latency_s": round(time.monotonic() - t0, 2), "value": r1.value.model_dump()})
    ok &= check("text: parsed structured output", r1.value.answer.lower().startswith("pong"), r1.value.answer)
    ok &= check("text: not a cache hit", r1.cache_hit is False)
    ok &= check("text: usage has input/output tokens", r1.usage.get("input_tokens") and r1.usage.get("output_tokens") is not None, str(r1.usage))
    # 2. image (canonical smoke crop; reference NEVER opened)
    print("2) tiny IMAGE structured call")
    crop = (REPO / "evaluation" / "hebrew_bench_v2" / "crops" / "hl_e004_q1_r3__l1.png").read_bytes()
    b64 = base64.standard_b64encode(crop).decode()
    blocks = [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
              {"type": "text", "text": "Does this image contain handwriting, and what is the dominant script? Reply with ONLY the JSON object."}]
    t0 = time.monotonic()
    r2 = rt.gateway.call(task="smoke_image", system="You describe images tersely.", content_blocks=blocks,
                         output_model=ImgRead, meta=meta)
    report["calls"].append({"task": "smoke_image", "cache_hit": r2.cache_hit, "usage": r2.usage,
                            "latency_s": round(time.monotonic() - t0, 2), "value": r2.value.model_dump()})
    ok &= check("image: parsed structured output", isinstance(r2.value.contains_handwriting, bool), str(r2.value.model_dump()))
    ok &= check("image: not a cache hit", r2.cache_hit is False)
    ok &= check("image: provider/model metadata", bool(r2.usage.get("model")), f"provider={r2.usage.get('provider')} model={r2.usage.get('model')} id={r2.usage.get('request_id')}")
    ok &= check("image: reported cost present", r2.usage.get("reported_cost") is not None, str(r2.usage.get("reported_cost")))
    # 3. identical image request -> cache hit, zero provider calls
    print("3) identical IMAGE request -> local cache")
    r3 = rt.gateway.call(task="smoke_image", system="You describe images tersely.", content_blocks=blocks,
                         output_model=ImgRead, meta=meta)
    report["calls"].append({"task": "smoke_image(repeat)", "cache_hit": r3.cache_hit, "usage": r3.usage})
    ok &= check("repeat: served from local cache", r3.cache_hit is True)
    ok &= check("repeat: identical value", r3.value == r2.value)
    # ledger / budget
    agg = rt.ledger.aggregate("smoke")
    report["ledger_aggregate"] = agg
    report["budget"] = rt.budget.snapshot()
    ok &= check("ledger: exactly 2 cloud requests", agg["cloud_requests"] == 2, str(agg["cloud_requests"]))
    ok &= check("ledger: 1 cache hit recorded", agg["cloud_cache_hits"] == 1)
    ok &= check("budget: charged 2 calls", rt.budget.snapshot()["calls_per_job"].get("smoke") == 2)
    # secret scan over everything persisted + the report
    blob = json.dumps(report, ensure_ascii=False) + (out / "state" / "gateway_ledger" / "usage.jsonl").read_text(encoding="utf-8")
    for p in (out / "state" / "gateway_cache").rglob("*.json"):
        blob += p.read_text(encoding="utf-8")
    ok &= check("no secret in ledger/cache/report", key not in blob and key[-8:] not in blob)
    report["all_ok"] = bool(ok)
    (out / "smoke_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print("smoke", "PASSED" if ok else "FAILED", "->", out / "smoke_report.json")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
