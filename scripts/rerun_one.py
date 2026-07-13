"""Re-grade ONE exam with the eval-batch protections, outside a batch.

- the model-visible document is an ANONYMIZED COPY (never the grade-bearing
  original filename/path);
- instructor red ink is masked exactly as in eval-batch (incl. the
  close-read's high-resolution re-render via the masked page loader);
- the result is labeled with the anonymized id only.

    python scripts/rerun_one.py --source test/003_70.pdf --anon-id exam-003 \
        --key sample_data/Exam_solution.pdf --out eval_out/exams/exam-003 \
        --base-url http://localhost:11434/v1 --model qwen3-vl:8b-instruct
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from autograder.backends import BackendConfig, create_backend
from autograder.cli import run_grade_pipeline
from autograder.ingest import load_pages
from autograder.masking import mask_pages


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--anon-id", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-url", default="http://localhost:11434/v1")
    ap.add_argument("--model", default="qwen3-vl:8b-instruct")
    ap.add_argument("--max-image-edge", type=int, default=1000)
    ap.add_argument("--survey-image-edge", type=int, default=640)
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--no-mask", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    anon_dir = Path(tempfile.mkdtemp(prefix="anon_exam_"))
    anon_path = anon_dir / f"{args.anon_id}.pdf"
    shutil.copyfile(args.source, anon_path)
    print(f"[rerun] anonymized copy: {anon_path.name}")

    backend = create_backend(
        BackendConfig(
            backend="openai",
            model=args.model,
            base_url=args.base_url,
            structured_mode="json_schema",
            temperature=0.0,
            timeout_s=args.timeout,
            max_tokens=16000,
        )
    )

    def loader(edge: int):
        loaded = load_pages(anon_path, edge)
        return loaded if args.no_mask else mask_pages(loaded)[0]

    ns = argparse.Namespace(
        key=args.key, rubric=None, resume=args.resume, version="auto",
        exam=str(anon_path), key_cache_dir=None, no_key_cache=False,
        variant_map=None, alignment_map=None,
    )
    try:
        result = run_grade_pipeline(
            ns, backend, Path(args.out), args.max_image_edge,
            exam_path=anon_path, exam_label=args.anon_id,
            pages=loader(args.max_image_edge),
            survey_image_edge=args.survey_image_edge,
            page_loader=loader,
        )
    finally:
        shutil.rmtree(anon_dir, ignore_errors=True)
    print(
        f"[rerun] {args.anon_id}: TOTAL {result.total_awarded:g}/{result.total_max:g} | "
        f"variant {result.detected_version} | review {len(result.needs_human_review)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
