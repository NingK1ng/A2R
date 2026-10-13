"""Units: the query and every candidate split into numbered sentences, deterministically.

No model rewrites the sources, so every reference to a source is a reference to its own words, and
two readings of the same case refer to the same units.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

QUERY = "query"  # source label of the query's own units, which are numbered Q1, Q2, ...

_MARKER = re.compile(r"^\s*(?:[-*•]+|#+|\d+[.)])\s+")  # list bullets, headings, enumerations
_END = re.compile(r"[.!?]+[\"')\]]*(?=\s+[\"'(\[]?[A-Z0-9])|[。！？]+")  # . ! ? before a capital; CJK stops
_ABBREVIATION = re.compile(r"(?:\b(?:[A-Za-z]\.){2,}|\b(?:Mr|Mrs|Ms|Dr|Prof|St|vs|cf|approx|Fig|No|Inc|Ltd|Jr|Sr|al)\.)$")


@dataclass(frozen=True)
class Unit:
    id: str
    source: str  # a candidate label, or QUERY
    text: str


def segment(text: str) -> list[str]:
    sentences = []
    for line in text.splitlines():
        line = _MARKER.sub("", line).strip()
        start = 0
        for end in _END.finditer(line):
            piece = line[start : end.end()].strip()
            if not _ABBREVIATION.search(piece):  # "U.S." or "e.g." does not end a sentence
                sentences.append(piece)
                start = end.end()
        if line[start:].strip():
            sentences.append(line[start:].strip())
    return sentences


def number(query: str, candidates: dict[str, str]) -> dict[str, Unit]:
    """Units of the query (Q1, Q2, ...) and of each candidate (A1, A2, ..., or label.1 if the label ends in a digit)."""
    if QUERY in candidates or "Q" in candidates:
        raise ValueError(f"the candidate labels {QUERY!r} and 'Q' are reserved for the query")
    units = {}
    for source, prefix, text in [(QUERY, "Q", query), *((label, label, text) for label, text in candidates.items())]:
        sep = "" if prefix[-1:].isalpha() else "."
        for n, sentence in enumerate(segment(text), 1):
            units[f"{prefix}{sep}{n}"] = Unit(f"{prefix}{sep}{n}", source, sentence)
    return units
