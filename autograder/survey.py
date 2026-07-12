"""Document-level survey pass over the student exam scan."""

from __future__ import annotations

from .ingest import PageImage, labeled_page_blocks
from .backends import VisionBackend
from .prompts import SURVEY_SYSTEM
from .schema import AnswerKey, ExamSurvey


def survey_exam(llm: VisionBackend, pages: list[PageImage], key: AnswerKey) -> ExamSurvey:
    blocks: list[dict] = [
        {
            "type": "text",
            "text": (
                "Exam structure from the answer key (for orientation only — it "
                "contains NO student information):\n"
                + "\n".join(
                    f"- Question {q.id}: {q.title} ({q.type}, {len(q.sub_items)} "
                    f"sub-items, {q.max_points} pts)"
                    + (f" — answers expected in: {q.answer_source}" if q.answer_source else "")
                    for q in key.questions
                )
                + (
                    f"\nExam versions that exist: {', '.join(key.versions)}"
                    if key.versions != ["default"]
                    else ""
                )
            ),
        }
    ]
    blocks.extend(labeled_page_blocks(pages))
    blocks.append(
        {
            "type": "text",
            "text": "Produce the document survey now.",
        }
    )
    return llm.parse(
        system=SURVEY_SYSTEM,
        content_blocks=blocks,
        output_model=ExamSurvey,
        max_tokens=16000,
    )
