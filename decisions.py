"""Rule-based decision extraction: flags sentences containing decision markers."""

import re

MARKERS = (
    "decided", "decision", "agreed", "we will", "let's", "approved",
    "finalized", "confirmed", "will start", "will use",
)


def extract_decisions(text: str, limit: int = 20) -> list[str]:
    out, seen = [], set()
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", text or ""):
        sentence = sentence.strip()
        if len(sentence) < 10:
            continue

        low = sentence.lower()
        if low in seen or not any(marker in low for marker in MARKERS):
            continue

        seen.add(low)
        out.append(sentence)

    return out[:limit]
