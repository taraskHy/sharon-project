"""Phase-5 integration diagnosis for Unlimited-OCR — ONE sanity image only.

The first sanity inference returned an empty string under the long
literal-transcription instruction prompt. This script probes DOCUMENTED
prompt syntaxes (README/code-comment task prompts) plus syntax-adapted
variants of the same uniform literal-transcription instruction, on the
single frozen sanity image, to find the invocation the model actually
follows. references.json is NEVER read; output goes to
evaluation/unlimited_ocr/diag_prompts.json.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "evaluation" / "hebrew_bench_v2"
OUT = REPO / "evaluation" / "unlimited_ocr"
SANITY_ITEM = "hl_e006_q1_r3__l1"

INSTR = (
    "Transcribe exactly the handwritten text visible in this image. "
    "Preserve Hebrew, English, numbers, symbols, formulas, spelling "
    "mistakes, and order as written. Do not answer the question, explain, "
    "correct spelling, normalize terminology, or infer missing content. "
    "Return only the transcription."
)

VARIANTS = {
    "readme_document_parsing": "<image>document parsing.",
    "code_free_ocr": "<image>\nFree OCR. ",
    "code_extract_text": "<image>\nExtract the text in the image. ",
    "instr_newline": f"<image>\n{INSTR}",
    "instr_original": f"<image>{INSTR}",
}


def main() -> int:
    snapshot = sys.argv[1]
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(snapshot, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        snapshot, trust_remote_code=True, use_safetensors=True,
        torch_dtype=torch.bfloat16, attn_implementation="eager",
    ).eval().cuda()

    img = BENCH / "crops" / f"{SANITY_ITEM}.png"
    scratch = OUT / "diag_scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, prompt in VARIANTS.items():
        t0 = time.monotonic()
        try:
            with torch.no_grad():
                raw = model.infer(
                    tokenizer, prompt=prompt, image_file=str(img),
                    output_path=str(scratch), base_size=1024, image_size=640,
                    crop_mode=True, save_results=False, eval_mode=True,
                    max_length=32768, no_repeat_ngram_size=35, ngram_window=128,
                )
        except Exception as e:  # noqa: BLE001
            raw = None
            results[name] = {"prompt": prompt, "error": f"{type(e).__name__}: {e}"}
            print(f"[{name}] ERROR {e}")
            continue
        dt = round(time.monotonic() - t0, 2)
        results[name] = {"prompt": prompt, "raw": raw, "latency_s": dt}
        print(f"[{name}] {dt}s -> {raw!r}")
    (OUT / "diag_prompts.json").write_text(
        json.dumps({"item": SANITY_ITEM, "results": results},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
