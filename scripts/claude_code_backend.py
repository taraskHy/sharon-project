"""EXPERIMENTAL Claude-Code subscription backend for the Claude Vision
assisted-annotation benchmark (evaluation/claude_candidates/PROTOCOL.md).

Uses the locally installed `claude` CLI (`claude -p`, Claude Max
subscription auth) instead of the separately billed Anthropic API. The
API backend (scripts/claude_candidates_run.py) is left unchanged and
unused. This backend REFUSES to run if ANTHROPIC_API_KEY is set, so a
run can never silently bill API credits.

Isolation per invocation (fresh, stateless):
- a throwaway sandbox directory holds ONE file: the anonymized line
  crop, copied to the neutral name crop.png (no writer/question/row in
  the filename);
- `claude -p` runs with that sandbox as cwd, a fresh session, tool
  allowlist limited to reading that single file, and everything else
  disallowed — no repository files, ground truth, grades, identifiers,
  keys, rubrics, or instructor ink are reachable or included;
- the full stream-json event log is saved BEFORE any evaluation.

    # one-crop smoke test (the only thing run without owner approval)
    .venv/Scripts/python.exe scripts/claude_code_backend.py smoke

    # 30-line benchmark — REQUIRES explicit owner approval flag
    .venv/Scripts/python.exe scripts/claude_code_backend.py generate \
        --config claude_line --owner-approved
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from scripts.claude_candidates_run import (  # noqa: E402  (fixed texts reused)
    B_CONTEXT_TEXT, INSTRUCTION, load_split_index,
)

ROOT = Path("evaluation/htr_pilot")
OUT = Path("evaluation/claude_candidates")
PROMPT_VERSION = "claude_htr_v1_cli"
CONFIGS = ("claude_line", "claude_line_cell")

SMOKE_PROMPT = (
    "An image file named crop.png is in the current working directory. "
    "Use the Read tool to view it, then output the transcription.\n"
    + INSTRUCTION
)


def find_cli() -> str:
    """Locate the installed claude CLI (desktop-app bundle or PATH)."""
    exe = shutil.which("claude")
    if exe:
        return exe
    base = Path(os.environ.get("APPDATA", "")) / "Claude" / "claude-code"
    if base.exists():
        versions = sorted((d for d in base.iterdir()
                           if (d / "claude.exe").exists()),
                          key=lambda d: [int(x) for x in d.name.split(".")
                                         if x.isdigit()])
        if versions:
            return str(versions[-1] / "claude.exe")
    raise FileNotFoundError("claude CLI not found (PATH or desktop bundle)")


def refuse_if_api_key() -> None:
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        print("REFUSING: ANTHROPIC_API_KEY/AUTH_TOKEN is set — this backend "
              "is subscription-only (owner instruction: no separately "
              "billed API usage). Unset it and retry.")
        raise SystemExit(2)


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def run_one(cli: str, images: list[Path], prompt: str, model: str | None,
            timeout_s: int = 300) -> dict:
    """One fresh stateless `claude -p` call in a throwaway sandbox that
    contains only neutral-named copies of `images`. Returns raw events +
    parsed analysis; writes nothing outside the sandbox."""
    sandbox = Path(tempfile.mkdtemp(prefix="claude_htr_"))
    names = []
    try:
        for i, src in enumerate(images):
            name = "crop.png" if len(images) == 1 else f"img{i + 1}.png"
            (sandbox / name).write_bytes(src.read_bytes())
            names.append(name)
        allow = [f"Read({n})" for n in names] + [f"Read(./{n})" for n in names]
        cmd = [cli, "-p", prompt,
               "--output-format", "stream-json", "--verbose",
               "--allowedTools", ",".join(allow),
               "--disallowedTools",
               "Bash,Write,Edit,WebFetch,WebSearch,Glob,Grep,Task,NotebookEdit",
               "--max-turns", "4"]
        if model:
            cmd += ["--model", model]
        env = {k: v for k, v in os.environ.items()
               if k not in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")}
        t0 = time.monotonic()
        proc = subprocess.run(cmd, cwd=sandbox, capture_output=True,
                              text=True, encoding="utf-8", timeout=timeout_s,
                              env=env)
        wall = round(time.monotonic() - t0, 1)
        events = []
        for line in (proc.stdout or "").splitlines():
            line = line.strip()
            if line.startswith("{"):
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        init = next((e for e in events if e.get("type") == "system"
                     and e.get("subtype") == "init"), {})
        result = next((e for e in events if e.get("type") == "result"), {})
        reads = []
        for e in events:
            if e.get("type") == "assistant":
                for b in (e.get("message") or {}).get("content", []):
                    if (isinstance(b, dict) and b.get("type") == "tool_use"
                            and b.get("name") == "Read"):
                        reads.append(b.get("input", {}).get("file_path", ""))
        image_read = any(any(n in r for n in names) for r in reads)
        return {
            "exit_code": proc.returncode, "wall_s": wall,
            "sandbox_files": names,
            "cli_version": init.get("version") if isinstance(init, dict) else None,
            "session_id": init.get("session_id"),
            "api_key_source": init.get("apiKeySource"),
            "model": init.get("model") or result.get("model"),
            "read_tool_calls": reads, "image_read": image_read,
            "result_text": (result.get("result") or "").strip(),
            "is_error": result.get("is_error"),
            "usage": result.get("usage"),
            "total_cost_usd_api_equivalent": result.get("total_cost_usd"),
            "num_turns": result.get("num_turns"),
            "stderr_tail": (proc.stderr or "")[-800:],
            "events": events,
        }
    finally:
        shutil.rmtree(sandbox, ignore_errors=True)


def cmd_smoke(args) -> int:
    refuse_if_api_key()
    cli = find_cli()
    ids = json.loads((OUT / "claude_bench_ids.json")
                     .read_text(encoding="utf-8"))["ids"]
    sid = ids[0]
    index = load_split_index(ROOT)
    src = ROOT / index[sid]["images"]["line"]
    print(f"cli: {cli}\nsmoke sample: {sid} (sent as crop.png, "
          f"sha256 {sha256(src.read_bytes())[:16]}…)")
    r = run_one(cli, [src], SMOKE_PROMPT, args.model)
    outdir = OUT / "claude_code_smoke"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "raw_stream.jsonl").write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in r["events"]) + "\n",
        encoding="utf-8")
    analysis = {k: v for k, v in r.items() if k != "events"}
    analysis.update({
        "sample_id": sid, "image_sha256": sha256(src.read_bytes()),
        "prompt_version": PROMPT_VERSION, "verified": False,
        "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })
    (outdir / "analysis.json").write_text(
        json.dumps(analysis, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: analysis[k] for k in
                      ("exit_code", "api_key_source", "model", "image_read",
                       "read_tool_calls", "num_turns", "usage",
                       "total_cost_usd_api_equivalent", "wall_s", "is_error")},
                     ensure_ascii=False, indent=1))
    print("\n--- transcription (UNVERIFIED AI CANDIDATE) ---")
    print(analysis["result_text"] or "(empty)")
    return 0 if (r["exit_code"] == 0 and r["image_read"]
                 and r["result_text"]) else 1


def load_ids(ids_file: str) -> list[str]:
    d = json.loads((OUT / (ids_file or "claude_bench_ids.json"))
                   .read_text(encoding="utf-8"))
    if "ids" in d:
        return d["ids"]
    return list(d["clear"]) + list(d["difficult"])  # early10 shape


def cmd_generate(args) -> int:
    refuse_if_api_key()
    if not args.owner_approved:
        print("REFUSING: benchmark generation runs only with explicit owner "
              "approval (pass --owner-approved).")
        return 2
    cli = find_cli()
    ids = load_ids(args.ids_file)
    index = load_split_index(ROOT)
    outdir = OUT / "outputs" / args.config
    outdir.mkdir(parents=True, exist_ok=True)
    for sid in ids:
        target = outdir / f"{sid}.json"
        if target.exists():
            continue
        s = index[sid]
        if args.config == "claude_line":
            images = [ROOT / s["images"]["line"]]
            prompt = SMOKE_PROMPT
        else:
            images = [ROOT / s["images"]["cell_clean"],
                      ROOT / s["images"]["line"]]
            prompt = ("Two image files are in the current working directory: "
                      "img1.png and img2.png. Use the Read tool to view both. "
                      + B_CONTEXT_TEXT.replace("first image", "img1.png")
                      .replace("second image", "img2.png") + "\n" + INSTRUCTION)
        r = run_one(cli, images, prompt, args.model)
        images_meta = {}
        for i, p in enumerate(images):
            key = "line" if p.name.endswith(tuple(
                f"l{j}.png" for j in range(1, 9))) else "cell_clean"
            images_meta[key] = {"path": str(p.relative_to(ROOT)),
                                "sha256": sha256(p.read_bytes())}
        target.write_text(json.dumps({
            "sample_id": sid, "config": args.config,
            "backend": "claude_code_subscription",
            "input_type": ("line_crop" if args.config == "claude_line"
                           else "line_crop+cell_context"),
            "model": r["model"], "prompt_version": PROMPT_VERSION,
            "api_key_source": r["api_key_source"],
            "image_read": r["image_read"],
            "read_tool_calls": r["read_tool_calls"],
            "images": images_meta,
            "raw_content": r["events"],
            "stop_reason": None, "usage": r["usage"],
            "candidate": r["result_text"] or None, "confidence": None,
            "error": (None if r["exit_code"] == 0 and r["image_read"]
                      and not r["is_error"] else
                      f"exit={r['exit_code']} image_read={r['image_read']} "
                      f"is_error={r['is_error']}"),
            "latency_s": r["wall_s"], "verified": False,
            "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[{args.config}] {sid}: {r['wall_s']}s "
              f"image_read={r['image_read']}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sm = sub.add_parser("smoke")
    sm.add_argument("--model", default="opus")
    ge = sub.add_parser("generate")
    ge.add_argument("--config", choices=CONFIGS, required=True)
    ge.add_argument("--model", default="opus")
    ge.add_argument("--owner-approved", action="store_true")
    ge.add_argument("--ids-file", default="",
                    help="ids file under evaluation/claude_candidates "
                         "(default: the full 30-line claude_bench_ids.json)")
    args = ap.parse_args()
    return {"smoke": cmd_smoke, "generate": cmd_generate}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
