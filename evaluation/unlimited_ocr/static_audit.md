# Static audit — baidu/Unlimited-OCR trust_remote_code files

Revision audited: `07dea832e22aefee32ad281d4b80551282e1c168` (local snapshot,
byte sizes identical to the HF API listing). Files: `modeling_unlimitedocr.py`
(53,431 B), `modeling_deepseekv2.py` (90,162 B), `deepencoder.py` (38,008 B),
`configuration_deepseek_v2.py` (10,720 B), `conversation.py` (9,253 B).
Method: pattern sweeps (subprocess/shell/registry/network/eval/exec/pickle/
env-vars/credentials) over the downloaded files + manual read of every hit
in context + full read of the `infer()` execution path.

## Findings

| Class | Result |
|---|---|
| subprocess / os.system / shell / PowerShell / cmd | **None** |
| Registry / ctypes / process termination / shutil deletion | **None** |
| Sockets / urllib / httpx / actual network calls | **None**. All URL strings are attribution comments (detectron2, ConvNeXt, FastChat, arXiv). `import requests` exists at module top but `requests` is never called anywhere — dead import (must merely be installed). |
| Environment/credential access (os.environ, getenv, expanduser, home, api_key, token) | **None** (only tokenizer `stop_token_ids` fields match the word "token") |
| pickle / torch.load | `torch.load(checkpoint)` exists in `deepencoder.py:_build_sam` but only under `if checkpoint is not None`; the model constructor calls `build_sam_vit_b()` / `build_clip_l()` with **no checkpoint** (modeling_unlimitedocr.py:437-438) → unreachable. Weights load via safetensors through transformers. |
| torch.compile | Only inside `build_sam_fast_vit_b`, which nothing in the model calls → unreachable. |
| eval()/exec() | `eval()` appears 8×, ALL on **model-generated output strings** (det-coordinate captures, geometry 'Line' dicts), and ALL inside the `save_results=True` branch of `infer()` (line 1069+) or functions called only from it (`draw_bounding_boxes` ← `process_image_with_refs`). With `save_results=False` (our configuration) these are unreachable. Noted as sloppy upstream practice, not hostile behavior; a crafted image could in principle steer eval input only if save_results were enabled. We keep it disabled. |
| Import-time side effects | None beyond imports. `disable_torch_init()` monkey-patches `torch.nn.Linear/LayerNorm.reset_parameters` (a known load-latency hack, process-local, called inside `infer`). |
| Filesystem writes | Only under caller-supplied `output_path` (`os.makedirs(output_path)` + `{output_path}/images` unconditional at infer() entry; result.md/jpg writes gated by save_results). No writes outside it. |

## Execution-path facts recorded for the runner

- `infer(...)` **returns the decoded text only when `eval_mode=True`** (stop
  token stripped, no marker post-processing → true raw); otherwise it only
  streams/saves. Runner passes `eval_mode=True`, `save_results=False`.
- temperature 0.0 → `do_sample=False` (greedy). `no_repeat_ngram_size=35` +
  `ngram_window=128` → `SlidingWindowNoRepeatNgramProcessor(35, 128)`, the
  documented pairing. R-SWA ring window is taken from
  `config.sliding_window_size` automatically.
- Images < 640 px stay a single global 1024-padded view (crop_ratio [1,1]);
  larger crops get dynamic local views (gundam mode).

## Verdict

**PASS — clean enough for normal ML model execution** under our frozen
configuration (`save_results=False`, `eval_mode=True`, offline mode, pinned
revision). No unexplained system/network/destructive behavior in the
execution path.
