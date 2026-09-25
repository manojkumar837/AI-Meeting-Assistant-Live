"""Builds the end-of-meeting report and renders it as plain text."""

from datetime import datetime

from .action_items import extract_action_items
from .decisions import extract_decisions
from .inference import summarize
from .topics import extract_topics


def build_final_report(transcript: str, model_path: str, meeting_type: str = "General Meeting") -> dict:
    return {
        "meeting_type": meeting_type,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "executive_summary": summarize(transcript, model_path, 180, 50),
        "topics": extract_topics(transcript),
        "decisions": extract_decisions(transcript),
        "action_items": extract_action_items(transcript),
        "transcript": transcript,
    }


def report_to_text(report: dict) -> str:
    lines = [
        "=" * 60, "AI MEETING REPORT", "=" * 60, "",
        f"Meeting Type: {report['meeting_type']}",
        f"Generated: {report['generated_at']}", "",
        "EXECUTIVE SUMMARY", "-" * 60, report["executive_summary"], "",
        "KEY TOPICS", "-" * 60,
    ]
    lines += [f"• {t}" for t in report["topics"]] or ["• None detected"]

    lines += ["", "DECISIONS", "-" * 60]
    lines += [f"• {d}" for d in report["decisions"]] or ["• None detected"]

    lines += ["", "ACTION ITEMS", "-" * 60]
    if report["action_items"]:
        for i, item in enumerate(report["action_items"], 1):
            lines.append(
                f"{i}. {item.get('person') or 'Unassigned'} | {item['action']} | "
                f"Deadline: {item.get('deadline') or 'Not detected'}"
            )
    else:
        lines.append("• None detected")

    lines += ["", "FULL TRANSCRIPT", "-" * 60, report["transcript"], "", "=" * 60]
    return "\n".join(lines)
