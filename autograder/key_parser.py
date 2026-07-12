"""Parse the official answer key + rubric into a structured ``AnswerKey``."""

from __future__ import annotations

import json
from pathlib import Path

from .ingest import PageImage, labeled_page_blocks
from .backends import VisionBackend
from .prompts import KEY_PARSER_SYSTEM
from .schema import AnswerKey


def parse_answer_key(
    llm: VisionBackend,
    pages: list[PageImage],
    rubric_text: str | None = None,
) -> AnswerKey:
    """Run the key-parsing pass over the solution document.

    ``rubric_text`` is an optional separate rubric supplied by the user; when
    present it is appended so its rules take part in extraction.
    """
    blocks: list[dict] = list(labeled_page_blocks(pages))

    text_layer = "\n\n".join(
        f"--- Page {p.page_number} text layer ---\n{p.text}" for p in pages if p.text
    )
    if text_layer:
        blocks.append(
            {
                "type": "text",
                "text": (
                    "Embedded text layer of the document (colours/highlighting are "
                    "NOT visible here — read those from the images):\n\n" + text_layer
                ),
            }
        )
    if rubric_text:
        blocks.append(
            {
                "type": "text",
                "text": "Additional grading rubric supplied separately:\n\n" + rubric_text,
            }
        )
    blocks.append(
        {
            "type": "text",
            "text": "Produce the structured answer key for this exam now.",
        }
    )
    return llm.parse(
        system=KEY_PARSER_SYSTEM,
        content_blocks=blocks,
        output_model=AnswerKey,
        max_tokens=16000,
    )


def load_answer_key(path: str | Path) -> AnswerKey:
    return AnswerKey.model_validate_json(Path(path).read_text(encoding="utf-8"))


def save_answer_key(key: AnswerKey, path: str | Path) -> None:
    Path(path).write_text(
        json.dumps(key.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
