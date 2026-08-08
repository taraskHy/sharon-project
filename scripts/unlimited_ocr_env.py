"""Persist environment metadata for the isolated Unlimited-OCR runtime.

Run with .venv-unlimited. Writes evaluation/unlimited_ocr/env.json with
exact package versions, torch CUDA runtime, GPU and BF16 capability, so
the experiment is reproducible. Imports torch/transformers only — never
the model's remote code.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "evaluation" / "unlimited_ocr"


def main() -> int:
    import torch

    smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
         "--format=csv,noheader"],
        capture_output=True, text=True, check=False,
    ).stdout.strip()
    meta = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            name: version(name)
            for name in ("torch", "torchvision", "transformers", "pillow",
                         "matplotlib", "einops", "addict", "easydict",
                         "pymupdf", "psutil", "huggingface-hub", "tokenizers",
                         "safetensors", "numpy")
        },
        "torch_cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "device_capability": (torch.cuda.get_device_capability(0)
                              if torch.cuda.is_available() else None),
        "bf16_supported": (torch.cuda.is_bf16_supported()
                           if torch.cuda.is_available() else None),
        "nvidia_smi": smi,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "env.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(json.dumps(meta, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
