"""Writing a reading out as an answer.

Each claim becomes one sentence that names the sources holding it and the sources denying it, each
under its condition if it has one. The claims of a group of alternatives follow one another, every one
after the first introduced by "Alternatively". Apart from the claims and conditions of the reading,
the writer adds only source names and these few connectives, so different readings are written as
different answers and the reading can be recovered from the answer.
"""
from __future__ import annotations

import re

from .reading import Reading
from .units import QUERY

_LOWER = {"the", "a", "an", "this", "that", "these", "those", "it", "its", "there", "their", "our", "both", "each", "every", "all", "some", "any", "no", "most"}
_VERB = {"+": ("holds", "hold"), "-": ("denies", "deny")}  # singular, plural


def clause(text: str) -> str:
    """A claim or condition as a clause that can follow 'that' or 'if'."""
    text = text.strip().rstrip(".!?;:").strip()
    first, _, rest = text.partition(" ")
    if first.lower() in _LOWER and first[:1].isupper() and not first.isupper():
        text = first.lower() + (" " + rest if rest else "")
    return text


def condition(text: str) -> str:
    g = re.sub(r"^(?:only if|if|when|whenever|provided(?: that)?)\s+", "", clause(text), flags=re.I)
    return g if re.match(r"unless\b", g, re.I) else "if " + g


def names(sources: list[str]) -> str:
    """'Candidate A', 'Candidates A and B', 'the question and Candidate C', ..."""
    labels = [s for s in sources if s != QUERY]
    parts = ["the question"] if QUERY in sources else []
    if labels:
        series = labels[0] if len(labels) == 1 else ", ".join(labels[:-1]) + " and " + labels[-1]
        parts.append(("Candidate " if len(labels) == 1 else "Candidates ") + series)
    return " and ".join(parts)


def attributions(reading: Reading, claim: str) -> list[tuple[str, str | None, list[str]]]:
    """(polarity, condition, sources) for every distinct stance taken on the claim: unconditional
    stances first, holding before denying, otherwise in the order the stances appear."""
    groups: dict[tuple[str, str | None], list[str]] = {}
    for s in reading.stances:
        if s.claim == claim:
            groups.setdefault((s.polarity, s.condition), []).append(s.unit)
    ordered = sorted(groups.items(), key=lambda item: (item[0][1] is not None, item[0][0] == "-"))
    return [(polarity, cond, reading.sources(units)) for (polarity, cond), units in ordered]


def write(reading: Reading) -> str:
    sentences = []
    for issue in reading.issues:
        for n, claim in enumerate(issue):
            parts = []
            for polarity, cond, sources in attributions(reading, claim):
                what = f"that {clause(reading.claims[claim])}" if not parts else "this"
                parts.append(f"{names(sources)} {_VERB[polarity][len(sources) > 1]} {what}" + (f" {condition(cond)}" if cond else ""))
            text = "; ".join(parts)
            sentences.append(f"Alternatively, {text}." if n else f"{text[0].upper()}{text[1:]}.")
    return " ".join(sentences)
