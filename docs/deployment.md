# Deployment

The application requires **no Anthropic key, no paid proprietary API, and no
specific vendor**: any OpenAI-compatible server serving an open-weight
vision-language model works via `--backend openai`. Three deployment shapes
are supported; switching between them is configuration only.

## 1. Local (Windows/Linux/macOS) via Ollama — development & small volumes

```powershell
winget install Ollama.Ollama          # or download from https://ollama.com
ollama pull qwen3-vl:8b               # ~6 GB, Apache-2.0 weights
autograder doctor --backend openai --base-url http://localhost:11434/v1 --model qwen3-vl:8b
autograder grade --backend openai --base-url http://localhost:11434/v1 --model qwen3-vl:8b `
    --exam sample_data/student_exam.pdf --key sample_data/Exam_solution.pdf --out out `
    --max-image-edge 1600 --timeout 1800
```

Hardware guidance:

| Setup | What runs | Expectation |
|---|---|---|
| CPU-only laptop (e.g. the dev machine: Core Ultra 7, 63 GB RAM, no discrete GPU) | Q4 8B VLM | Works, but vision prefill on 13-page exams is **slow** (minutes per call, potentially hours per exam). Good for smoke tests, not batch grading |
| 1× 24 GB GPU (RTX 3090/4090, A10, L4) | 8B bf16 or AWQ/FP8; 14B FP8 | Comfortable; recommended minimum for real use |
| 1× 48–80 GB GPU (A100/H100/L40S) | 32B, MoE 30B-A3B, 24B bf16 | Headroom for the stronger models if 8B accuracy is insufficient |
| Disk | 6–20 GB per model + ~1 GB app/venv | |

Ollama notes: set a large context (`OLLAMA_CONTEXT_LENGTH` or request-level
`num_ctx`) — multi-image exam calls overflow small default contexts, which
manifests as silently degraded output rather than an error; keep
`--concurrency 1`; prefer `--structured-mode json_schema` (Ollama constrained
decoding), fall back to `json_object` if a model fights the grammar.

## 2. University Linux GPU server via vLLM — production path

```bash
pip install vllm
vllm serve Qwen/Qwen3-VL-8B-Instruct --host 0.0.0.0 --port 8000 \
    --max-model-len 32768 --limit-mm-per-prompt image=16
# then, from any machine:
autograder eval-batch --split validation \
    --backend openai --base-url http://gpu-server:8000/v1 --model Qwen/Qwen3-VL-8B-Instruct \
    --key sample_data/Exam_solution.pdf --out eval_out
```

- vLLM's OpenAI-compatible endpoint supports multi-image chat requests and
  `response_format={"type": "json_schema"}` constrained decoding.
- Student data never leaves the university network.
- If the server requires a token, export it and pass `--api-key-env NAME`.
- HF TGI and llama.cpp's `llama-server` are drop-in alternatives (same
  `--backend openai`); TGI's upstream development was archived in 2025, so
  vLLM is the recommended primary.

## 3. Free hosted APIs for open models — fallback / no-hardware option

See docs/model-comparison.md §hosted APIs for the surveyed providers, quotas,
and data-use policies (with dates — free tiers change). Ground rules:

- the pipeline works with any of them via
  `--backend openai --base-url <provider>/v1 --api-key-env <ENV>`;
- **check the provider's data-retention/training policy before sending real
  student scans** (docs/privacy-and-leakage.md) — most free tiers are only
  appropriate for development on non-sensitive data;
- rate limits on free tiers make batch grading slow but feasible (an exam is
  ~6–10 requests with ~5–15 images total);
- do not assume any free tier will remain free: the architecture deliberately
  keeps a self-hosted path so the university is never forced to redesign.

## Operational commands

- `autograder doctor [backend flags]` — reachability + model availability.
- All timeouts, retries, image resolution, generation parameters, and
  concurrency are configurable (CLI or TOML; `autograder grade --help`).
- Results record the exact backend/model/config (`backend_info`), and
  `--resume` refuses to reuse stages when any input or configuration changed.
