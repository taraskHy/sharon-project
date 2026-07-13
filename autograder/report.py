"""Render an ``ExamResult`` as a human-readable Markdown report."""

from __future__ import annotations

from .schema import ExamResult


def _fmt(x: float) -> str:
    return f"{x:g}"


def render_markdown(result: ExamResult) -> str:
    lines: list[str] = []
    lines.append(f"# Grading report — {result.exam_file}")
    lines.append("")
    lines.append(f"**Total: {_fmt(result.total_awarded)} / {_fmt(result.total_max)}**")
    lines.append("")
    lines.append(f"- Graded at: {result.graded_at}")
    lines.append(f"- Model: {result.model}")
    lines.append(f"- Exam version: {result.detected_version} ({result.version_detection})")
    if result.variant_detection:
        vd = result.variant_detection
        lines.append(
            f"- Variant marker: {vd.get('marker_kind')} "
            f"{vd.get('matched_marker') or '(unmatched: ' + str(vd.get('marker_seen')) + ')'}"
            f" on page {vd.get('page')} ({vd.get('page_region')}); "
            f"confident={vd.get('confident')}; mapping: {vd.get('mapping_source')}; "
            f"alignment: {vd.get('question_alignment')}"
        )
    lines.append("")

    lines.append("## Score breakdown")
    lines.append("")
    lines.append("| Question | Type | Awarded | Max | Notes |")
    lines.append("|---|---|---:|---:|---|")
    for q in result.questions:
        lines.append(
            f"| {q.question_id} | {q.question_type} | {_fmt(q.points_awarded)} | "
            f"{_fmt(q.points_max)} | {q.summary} |"
        )
    lines.append("")

    for q in result.questions:
        lines.append(f"## Question {q.question_id} — {_fmt(q.points_awarded)}/{_fmt(q.points_max)}")
        lines.append("")
        lines.append("| Item | Answer | Correct | Sel. pts | Expl. pts | Total | Reason |")
        lines.append("|---|---|---|---:|---:|---:|---|")
        for s in q.sub_results:
            answer = s.student_answer or ("—" if s.status == "unanswered" else "?")
            correct = {True: "yes", False: "no", None: "n/a"}[s.selection_correct]
            flag = " ⚠" if s.needs_review else ""
            reason = s.reason.replace("|", "\\|")
            lines.append(
                f"| {s.sub_item_id}{flag} | {answer} | {correct} | "
                f"{_fmt(s.points_selection)} | {_fmt(s.points_explanation)} | "
                f"{_fmt(s.points_total)}/{_fmt(s.points_max)} | {reason} |"
            )
        lines.append("")
        with_expl = [s for s in q.sub_results if s.explanation_transcription]
        if with_expl:
            lines.append("<details><summary>Transcribed explanations</summary>")
            lines.append("")
            for s in with_expl:
                verdict = s.explanation_evaluation.verdict if s.explanation_evaluation else "—"
                lines.append(f"- **{s.sub_item_id}** ({verdict}): {s.explanation_transcription}")
            lines.append("")
            lines.append("</details>")
            lines.append("")

    if result.unanswered:
        lines.append("## Unanswered")
        lines.append("")
        for item in result.unanswered:
            lines.append(f"- Q{item.question_id} item {item.sub_item_id}")
        lines.append("")

    if result.needs_human_review:
        lines.append("## Needs human review")
        lines.append("")
        for item in result.needs_human_review:
            lines.append(f"- Q{item.question_id} item {item.sub_item_id}: {item.reason}")
        lines.append("")

    if result.mark_interpretations:
        lines.append("## How markings were interpreted")
        lines.append("")
        for note in result.mark_interpretations:
            lines.append(f"- {note}")
        lines.append("")

    return "\n".join(lines)
