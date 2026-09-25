"""Lightweight keyword-frequency topic extraction (no ML model needed)."""

import re
from collections import Counter

STOPWORDS = set("""
about after again against also because before being between could during
from have having into more most other over same should some such than
that their there these they this through under very what when where
which while with would your will were been them then meeting today
good morning okay yes need needs
""".split())


def extract_topics(text: str, top_n: int = 8) -> list[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9_-]{3,}", (text or "").lower())
    counts = Counter(w for w in words if w not in STOPWORDS)
    return [w.replace("_", " ").title() for w, _ in counts.most_common(top_n)]
