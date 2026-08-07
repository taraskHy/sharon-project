"""DeepSeek-OCR-2 CPU feasibility probe (Mission 2).

Owner-approved scope: OFFICIAL deepseek-ai model source only, exact
revision pinned, executed inside the isolated .venv-dsocr environment,
no global machine changes. This script records exactly what was executed
(model id, revision hash, remote-code files and their SHA256) into the
probe report before any inference.

The official stack targets CUDA + flash-attn; this probe tests whether the
official custom code can run CPU-only with eager attention. Either outcome
is a documented result.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "evaluation" / "m2_deepseek_probe.json"

MODEL_ID = "deepseek-ai/DeepSeek-OCR-2"
# Pin resolved at probe time from the main branch and recorded below.

report: dict = {"model_id": MODEL_ID, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                "isolated_env": sys.prefix, "steps": []}


def step(name, **kw):
    entry = {"step": name, **kw}
    report["steps"].append(entry)
    print(f"[{name}] " + json.dumps(kw, ensure_ascii=False, default=str)[:300])
    OUT.write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str),
                   encoding="utf-8")


def main() -> int:
    from huggingface_hub import HfApi, snapshot_download

    api = HfApi()
    info = api.model_info(MODEL_ID)
    revision = info.sha
    step("resolve_revision", revision=revision, last_modified=info.last_modified)

    local = snapshot_download(MODEL_ID, revision=revision)
    step("snapshot_downloaded", path=local)

    # Record every remote-code file and its hash BEFORE executing any of it.
    code_files = {}
    for f in sorted(Path(local).glob("*.py")):
        code_files[f.name] = hashlib.sha256(f.read_bytes()).hexdigest()
    step("remote_code_inventory", files=code_files)
    weights = sorted(p.name for p in Path(local).glob("*.safetensors"))
    pickles = sorted(p.name for p in Path(local).glob("*.bin"))
    step("weights_inventory", safetensors=weights, pickle_bins=pickles)
    if not weights and pickles:
        step("abort", reason="no safetensors weights; pickle-only is not approved")
        return 2

    import torch
    from transformers import AutoModel, AutoTokenizer

    step("load_attempt", dtype="float32", attn="eager", device="cpu")
    try:
        tok = AutoTokenizer.from_pretrained(local, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            local,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            attn_implementation="eager",
            use_safetensors=True,
        )
        model = model.eval().to("cpu")
        step("load_ok", params_m=round(sum(p.numel() for p in model.parameters()) / 1e6))
    except Exception as e:  # noqa: BLE001
        step("load_failed", error=f"{type(e).__name__}: {e}")
        report["verdict"] = "LOAD FAILED on CPU/eager with official code"
        OUT.write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str),
                       encoding="utf-8")
        return 1

    # one benchmark crop, official infer entrypoint per model card
    crop = REPO / "evaluation" / "hebrew_bench_v2" / "crops" / "pr_docA_p2_b1.png"
    t0 = time.monotonic()
    try:
        if hasattr(model, "infer"):
            res = model.infer(tok, prompt="<image>\nFree OCR.",
                              image_file=str(crop), output_path=str(REPO / "evaluation"),
                              base_size=1024, image_size=640, crop_mode=True,
                              save_results=False, test_compress=False)
        else:
            res = "no infer() entrypoint; available: " + ", ".join(
                m for m in dir(model) if not m.startswith("_"))[:400]
        step("inference", seconds=round(time.monotonic() - t0, 1), result=str(res)[:500])
        report["verdict"] = "RUNS on CPU (see inference step for output/timing)"
    except Exception as e:  # noqa: BLE001
        step("inference_failed", seconds=round(time.monotonic() - t0, 1),
             error=f"{type(e).__name__}: {e}")
        report["verdict"] = "loads but inference FAILED on CPU"
    OUT.write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str),
                   encoding="utf-8")
    print("verdict:", report["verdict"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
