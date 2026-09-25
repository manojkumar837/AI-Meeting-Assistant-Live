"""Rule-based action item extraction: flags sentences with action language
and pulls out an optional owner and deadline."""

import re

ACTION_WORDS = (
    "will", "should", "need to", "needs to", "must", "has to", "have to",
    "assigned", "responsible", "please", "todo", "to do", "complete",
    "finish", "prepare", "send", "review", "fix", "update", "follow up",
    "schedule", "create", "handle", "deliver",
)

DEADLINE_RE = re.compile(
    r"\b(?:by|before|on|until|due(?:\s+by)?)\s+([A-Za-z0-9 ,:/-]{2,40}?)(?=[.!?,;\n]|$)",
    re.I,
)
OWNER_RE = re.compile(r"([A-Z][a-z]{1,20})\s+(?:will|should|must|needs?|is responsible)")


def extract_action_items(text: str, limit: int = 30) -> list[dict]:
    out, seen = [], set()
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", text or ""):
        sentence = sentence.strip()
        if len(sentence) < 8:
            continue

        low = sentence.lower()
        if low in seen or not any(word in low for word in ACTION_WORDS):
            continue

        seen.add(low)
        deadline_match = DEADLINE_RE.search(sentence)
        owner_match = OWNER_RE.match(sentence)

        out.append({
            "person": owner_match.group(1) if owner_match else "",
            "action": sentence,
            "deadline": deadline_match.group(1).strip(" .,") if deadline_match else "",
        })

    return out[:limit]
