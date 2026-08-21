"""Persistent course-material store + local RAG index (experimental arm).

Lifecycle: a lecturer creates a course once, uploads course summaries /
material (PDF / TXT / Markdown / DOCX), and the server parses, chunks and
embeds them into a persistent per-course index that later grading batches
reuse — material is never re-uploaded per batch.

Layout (root configurable via GRADER_COURSES_DIR, default ./courses):

    courses/<course_id>/
        sources/            original uploaded files
        parsed/<hash>.json  deterministic text extraction per source
        chunks/chunks.jsonl one chunk per line (id, text, source, page, section)
        rag_index/
            embeddings.npy      float32 [n_chunks, dim]
            index_manifest.json source hashes, embed model, chunk config,
                                n_chunks, build timestamp, config hash

Safety: files whose names look like answer keys / rubrics are REFUSED at
ingestion, and extracted CONTENT is screened with conservative answer-key/
rubric indicators — suspicious files are refused unless the operator
explicitly overrides after verifying them. The grading key must never leak
into OCR-repair retrieval. Embeddings default to a local Ollama model
(multilingual bge-m3); an ``embed_fn`` can be injected for tests.

Everything here is deterministic given the same sources + config: chunk
ids are content hashes, parsing has no randomness, and the index manifest
records every configuration knob so material or config changes invalidate
exactly the right course index.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional
from xml.etree import ElementTree

import numpy as np

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md", ".markdown", ".docx"}
# Filename gate (fast, obvious aliases). English tokens are bounded by
# non-letters so "keynote"/"monkey" stay allowed while "key_2024" or
# "grading_notes" are caught; Hebrew is matched as substrings.
KEY_LIKE = re.compile(
    r"answer[_\s-]*key|rubric|solution"
    r"|(?<![a-z])(?:references?|grading|grades?|answers?|keys?)(?![a-z])"
    r"|מחוון|פתרון|תשובות|מפתח|ציונים",
    re.IGNORECASE,
)
# Content gate (conservative — STRONG indicators only, so ordinary lecture
# notes never trip it): explicit key/rubric marker phrases, or a dense
# numbered option-letter list ("3. ב" style) typical of MC answer keys.
_CONTENT_MARKERS = re.compile(
    r"answer\s+key|grading\s+rubric|\brubric\b|מחוון|מפתח\s+תשובות"
    r"|פתרון\s+(?:מלא|רשמי)|טבלת\s+תשובות",
    re.IGNORECASE,
)
_ANSWER_LINE = re.compile(r"^\s*\d{1,3}\s*[.)\-:]?\s*[א-דA-Da-d]\s*$", re.MULTILINE)

DEFAULT_EMBED_MODEL = "bge-m3"
DEFAULT_CHUNK_CONFIG = {
    "target_chars": 600,
    "max_chars": 1100,
    "overlap_paragraphs": 1,
    "version": 1,
}


def courses_root() -> Path:
    return Path(os.environ.get("GRADER_COURSES_DIR", "courses"))


def course_dir(course_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_\-]{1,64}", course_id):
        raise ValueError(f"invalid course id {course_id!r}")
    return courses_root() / course_id


def create_course(course_id: str, name: str = "") -> Path:
    d = course_dir(course_id)
    for sub in ("sources", "parsed", "chunks", "rag_index"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    meta = d / "course.json"
    if not meta.exists():
        meta.write_text(json.dumps({
            "course_id": course_id, "name": name or course_id,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, ensure_ascii=False, indent=1), encoding="utf-8")
    return d


def list_courses() -> list[dict]:
    root = courses_root()
    if not root.exists():
        return []
    out = []
    for d in sorted(root.iterdir()):
        meta = d / "course.json"
        if meta.exists():
            out.append(json.loads(meta.read_text(encoding="utf-8")))
    return out


# --------------------------------------------------------------------------
# ingestion
# --------------------------------------------------------------------------


def content_suspicion(blocks: list[dict]) -> str | None:
    """Conservative screen of EXTRACTED text for answer-key/rubric material.

    Only strong indicators fire (marker phrases twice, or a dense numbered
    option-letter list) — normal lecture notes must never trip this."""
    text = "\n".join(b["text"] for b in blocks)
    markers = _CONTENT_MARKERS.findall(text)
    if len(markers) >= 2:
        return (f"content carries answer-key/rubric markers "
                f"({len(markers)}x, e.g. {markers[0]!r})")
    pairs = _ANSWER_LINE.findall(text)
    if len(pairs) >= 6:
        return (f"content contains a dense numbered answer-letter list "
                f"({len(pairs)} lines shaped like '<n>. <letter>')")
    return None


def add_source(course_id: str, filename: str, data: bytes,
               allow_suspicious: bool = False) -> dict:
    """Store one uploaded file. Key/rubric-like NAMES are refused outright;
    suspicious CONTENT is refused unless the operator explicitly overrides
    (``allow_suspicious=True``) after verifying the file."""
    d = create_course(course_id)
    if Path(filename).suffix.lower() not in SUPPORTED_SUFFIXES:
        return {"stored": False, "filename": filename,
                "reason": f"unsupported type {Path(filename).suffix!r}"}
    if KEY_LIKE.search(filename):
        return {"stored": False, "filename": filename,
                "reason": "looks like an answer key / rubric — grading keys "
                          "must not enter the course-material corpus"}
    safe = re.sub(r"[^\w֐-׿.\- ]", "_", Path(filename).name)
    target = d / "sources" / safe
    sha = hashlib.sha256(data).hexdigest()
    if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == sha:
        return {"stored": True, "filename": safe, "sha256": sha}
    target.write_bytes(data)
    try:
        reason = content_suspicion(parse_source(target)["blocks"])
    except Exception:  # noqa: BLE001 — unparseable files surface at build time
        reason = None
    if reason and not allow_suspicious:
        target.unlink()
        return {"stored": False, "filename": filename, "suspicious": True,
                "reason": f"{reason} — refused; verify the file and use the "
                          "operator override only if it is genuinely course "
                          "material"}
    out = {"stored": True, "filename": safe, "sha256": sha}
    if reason:
        out["suspicious_override"] = reason
        _record_override(d, safe, reason)
    return out


def remove_source(course_id: str, filename: str) -> bool:
    p = course_dir(course_id) / "sources" / Path(filename).name
    if p.exists():
        p.unlink()
        return True
    return False


def _record_override(course_dir_path: Path, filename: str, reason: str) -> None:
    """Persist an operator override into course.json — a corpus that contains
    operator-approved flagged material must be distinguishable from a clean
    one in every later audit (never just a transient UI warning)."""
    meta_path = course_dir_path / "course.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    overrides = [o for o in meta.get("suspicious_overrides", [])
                 if o.get("filename") != filename]
    overrides.append({"filename": filename, "reason": reason,
                      "ts": time.strftime("%Y-%m-%d %H:%M:%S")})
    meta["suspicious_overrides"] = overrides
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                         encoding="utf-8")


def _overridden_names(course_dir_path: Path) -> set[str]:
    meta_path = course_dir_path / "course.json"
    if not meta_path.exists():
        return set()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return {o.get("filename") for o in meta.get("suspicious_overrides", [])}


def _extract_docx(data: bytes) -> list[tuple[Optional[str], str]]:
    """(section_heading, paragraph) pairs from a .docx, no extra deps."""
    ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    out: list[tuple[Optional[str], str]] = []
    import io

    with zipfile.ZipFile(io.BytesIO(data)) as z:
        root = ElementTree.fromstring(z.read("word/document.xml"))
    section = None
    for para in root.iter(f"{ns}p"):
        style = para.find(f"{ns}pPr/{ns}pStyle")
        text = "".join(t.text or "" for t in para.iter(f"{ns}t")).strip()
        if not text:
            continue
        if style is not None and "Heading" in (style.get(f"{ns}val") or ""):
            section = text
            continue
        out.append((section, text))
    return out


def parse_source(path: Path) -> dict:
    """Deterministic text extraction with per-block metadata."""
    data = path.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    blocks: list[dict] = []
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        import fitz

        with fitz.open(stream=data, filetype="pdf") as doc:
            for pno, page in enumerate(doc, start=1):
                for b in page.get_text("blocks"):
                    text = (b[4] or "").strip()
                    if text:
                        blocks.append({"text": text, "page": pno, "section": None})
    elif suffix == ".docx":
        for section, text in _extract_docx(data):
            blocks.append({"text": text, "page": None, "section": section})
    else:  # txt / md
        text = data.decode("utf-8", errors="replace")
        section = None
        for para in re.split(r"\n\s*\n", text):
            para = para.strip()
            if not para:
                continue
            m = re.match(r"^(#{1,6})\s+(.+)$", para.splitlines()[0])
            if m:
                section = m.group(2).strip()
                body = "\n".join(para.splitlines()[1:]).strip()
                if body:
                    blocks.append({"text": body, "page": None, "section": section})
                continue
            blocks.append({"text": para, "page": None, "section": section})
    return {"source": path.name, "sha256": sha, "blocks": blocks}


# --------------------------------------------------------------------------
# chunking
# --------------------------------------------------------------------------


def chunk_parsed(parsed: dict, config: dict | None = None) -> list[dict]:
    """Heading/paragraph-aware chunking with limited overlap. Deterministic:
    chunk ids hash (source sha, block span, text)."""
    cfg = {**DEFAULT_CHUNK_CONFIG, **(config or {})}
    chunks: list[dict] = []
    buf: list[dict] = []
    buf_len = 0

    def flush():
        nonlocal buf, buf_len
        if not buf:
            return
        text = "\n".join(b["text"] for b in buf).strip()
        if len(text) < 25:  # ignore fragments
            buf, buf_len = [], 0
            return
        cid = hashlib.sha1(
            f"{parsed['sha256']}|{buf[0].get('_i')}|{text}".encode()
        ).hexdigest()[:16]
        chunks.append({
            "chunk_id": cid,
            "text": text,
            "source": parsed["source"],
            "source_sha256": parsed["sha256"],
            "page": buf[0].get("page"),
            "section": buf[0].get("section"),
        })
        overlap = buf[-cfg["overlap_paragraphs"]:] if cfg["overlap_paragraphs"] else []
        buf = list(overlap)
        buf_len = sum(len(b["text"]) for b in buf)

    prev_section = object()
    for i, b in enumerate(parsed["blocks"]):
        b = dict(b, _i=i)
        section_changed = b.get("section") != prev_section and prev_section is not object()
        if buf and (section_changed or buf_len + len(b["text"]) > cfg["max_chars"]):
            flush()
        buf.append(b)
        buf_len += len(b["text"])
        prev_section = b.get("section")
        if buf_len >= cfg["target_chars"]:
            flush()
    flush()
    return chunks


# --------------------------------------------------------------------------
# embeddings + index
# --------------------------------------------------------------------------


def ollama_embed_fn(model: str = DEFAULT_EMBED_MODEL,
                    base_url: str = "http://localhost:11434") -> Callable:
    import httpx

    def embed(texts: list[str]) -> np.ndarray:
        with httpx.Client(timeout=300.0) as client:
            resp = client.post(f"{base_url}/api/embed",
                               json={"model": model, "input": texts})
            resp.raise_for_status()
            return np.asarray(resp.json()["embeddings"], dtype=np.float32)

    embed.model_name = model  # type: ignore[attr-defined]
    return embed


def _config_hash(source_hashes: dict, embed_model: str, chunk_cfg: dict) -> str:
    payload = json.dumps({"sources": source_hashes, "embed": embed_model,
                          "chunking": chunk_cfg}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def index_status(course_id: str) -> dict:
    d = course_dir(course_id)
    manifest_p = d / "rag_index" / "index_manifest.json"
    sources = sorted((d / "sources").glob("*")) if (d / "sources").exists() else []
    src_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    status = {"course_id": course_id, "n_sources": len(sources),
              "sources": list(src_hashes), "indexed": False, "stale": False,
              "n_chunks": 0, "refused": []}
    if manifest_p.exists():
        m = json.loads(manifest_p.read_text(encoding="utf-8"))
        status.update({"indexed": True, "n_chunks": m.get("n_chunks", 0),
                       "built": m.get("built"), "embed_model": m.get("embed_model"),
                       "config_hash": m.get("config_hash")})
        current = _config_hash(src_hashes, m.get("embed_model", ""),
                               m.get("chunk_config", {}))
        status["stale"] = current != m.get("config_hash")
    return status


def build_index(course_id: str, embed_fn: Callable | None = None,
                chunk_config: dict | None = None) -> dict:
    """Parse -> chunk -> embed -> persist. Rebuilds only this course."""
    d = create_course(course_id)
    embed_fn = embed_fn or ollama_embed_fn()
    embed_model = getattr(embed_fn, "model_name", "injected")
    chunk_cfg = {**DEFAULT_CHUNK_CONFIG, **(chunk_config or {})}

    # Build-time re-screen: the add_source gates only cover the UI door.
    # Files placed in sources/ out-of-band (manual copy, sync tooling, a
    # crash between write and screen) must face the SAME gates here, or the
    # index silently ingests key/rubric material. Operator-overridden files
    # (persisted in course.json) are the only sanctioned exceptions.
    overridden = _overridden_names(d)
    excluded: list[dict] = []
    all_chunks: list[dict] = []
    src_hashes: dict[str, str] = {}
    for src in sorted((d / "sources").glob("*")):
        if src.suffix.lower() not in SUPPORTED_SUFFIXES:
            excluded.append({"file": src.name,
                             "reason": f"unsupported type {src.suffix!r}"})
            continue
        if KEY_LIKE.search(src.name):
            excluded.append({"file": src.name,
                             "reason": "key/rubric-like filename"})
            continue
        parsed = parse_source(src)
        reason = content_suspicion(parsed["blocks"])
        if reason and src.name not in overridden:
            excluded.append({"file": src.name, "reason": reason})
            continue
        src_hashes[src.name] = parsed["sha256"]
        (d / "parsed" / f"{parsed['sha256'][:16]}.json").write_text(
            json.dumps(parsed, ensure_ascii=False), encoding="utf-8")
        all_chunks.extend(chunk_parsed(parsed, chunk_cfg))
    if not all_chunks:
        raise ValueError(f"course {course_id!r} has no parseable material")

    with (d / "chunks" / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    vecs = embed_fn([c["text"] for c in all_chunks])
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / np.clip(norms, 1e-9, None)
    np.save(d / "rag_index" / "embeddings.npy", vecs.astype(np.float32))

    manifest = {
        "course_id": course_id,
        "source_hashes": src_hashes,
        "embed_model": embed_model,
        "chunk_config": chunk_cfg,
        "n_chunks": len(all_chunks),
        "dim": int(vecs.shape[1]),
        "built": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config_hash": _config_hash(src_hashes, embed_model, chunk_cfg),
        # Audit trail: what the build-time screen kept out, and which indexed
        # files entered only via a persisted operator override.
        "excluded_sources": excluded,
        "suspicious_overrides": sorted(overridden & set(src_hashes)),
    }
    (d / "rag_index" / "index_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    return manifest


def retrieve(course_id: str, query: str, top_k: int = 4,
             embed_fn: Callable | None = None) -> list[dict]:
    """Cosine top-k over the persisted course index."""
    d = course_dir(course_id)
    vec_p = d / "rag_index" / "embeddings.npy"
    if not vec_p.exists():
        return []
    embed_fn = embed_fn or ollama_embed_fn()
    chunks = [json.loads(l) for l in
              (d / "chunks" / "chunks.jsonl").read_text(encoding="utf-8").splitlines()]
    vecs = np.load(vec_p)
    q = embed_fn([query])[0]
    q = q / max(float(np.linalg.norm(q)), 1e-9)
    sims = vecs @ q
    order = np.argsort(-sims)[:top_k]
    return [{
        "chunk_id": chunks[i]["chunk_id"],
        "text": chunks[i]["text"],
        "source": chunks[i]["source"],
        "page": chunks[i].get("page"),
        "section": chunks[i].get("section"),
        "similarity": round(float(sims[i]), 4),
    } for i in order]
