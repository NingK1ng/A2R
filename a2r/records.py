"""Exact text differences, transmitted patches, and reconstructed source frames."""
from __future__ import annotations

from dataclasses import dataclass

from .units import QUERY, Unit

Span = tuple[str, int, int, int, int]


@dataclass(frozen=True)
class Error:
    prediction: tuple[str, ...]
    current: tuple[Unit, ...]
    spans: tuple[Span, ...]

    @property
    def changes(self) -> tuple[Span, ...]:
        return tuple(span for span in self.spans if span[0] != "equal")


@dataclass(frozen=True)
class Alignment:
    """A contextual relation between two nonempty, whole-unit intervals."""

    block: int
    old_start: int
    old_stop: int
    new_start: int
    new_stop: int
    old_to_new: bool
    new_to_old: bool

    @property
    def equivalent(self) -> bool:
        return self.old_to_new and self.new_to_old


@dataclass(frozen=True)
class Patch:
    start: int
    stop: int
    text: tuple[str, ...]

    def to_dict(self) -> dict:
        return {"start": self.start, "stop": self.stop, "text": list(self.text)}


@dataclass(frozen=True)
class Packet:
    source: str
    patches: tuple[Patch, ...]
    evidence: tuple[tuple[str, ...], ...]

    @classmethod
    def parse(cls, raw: dict, prediction: tuple[str, ...],
              units: dict[str, Unit]) -> Packet:
        if not isinstance(raw, dict) or set(raw) != {"source", "patches", "evidence"}:
            raise ValueError("packet must contain exactly source, patches, evidence")
        source = raw["source"]
        if not isinstance(source, str) or not source or source == QUERY:
            raise ValueError("packet source must be a nonempty candidate label")
        if not isinstance(raw["patches"], list) or not isinstance(raw["evidence"], list):
            raise ValueError("patches and evidence must be lists")
        patches, cursor, length = [], 0, len(prediction)
        for item in raw["patches"]:
            if not isinstance(item, dict) or set(item) != {"start", "stop", "text"}:
                raise ValueError("each patch needs exactly start, stop, text")
            start, stop, text = item["start"], item["stop"], item["text"]
            if type(start) is not int or type(stop) is not int or not 0 <= start <= stop <= len(prediction):
                raise ValueError("patch bounds must be integer prediction coordinates")
            if start < cursor or (patches and start == stop == patches[-1].start == patches[-1].stop):
                raise ValueError("patches must be ordered and nonoverlapping")
            if not isinstance(text, list) or any(not isinstance(line, str) or not line.strip() for line in text):
                raise ValueError("patch text must contain nonempty strings")
            if start == stop and not text:
                raise ValueError("an insertion cannot be empty")
            patches.append(Patch(start, stop, tuple(text)))
            cursor, length = stop, length + len(text) - (stop - start)
        if len(raw["evidence"]) != length:
            raise ValueError("evidence must cover every reconstructed text unit")
        evidence = []
        for refs in raw["evidence"]:
            if not isinstance(refs, list) or not refs:
                raise ValueError("each reconstructed unit needs current-source evidence")
            if any(not isinstance(ref, str) or ref not in units or units[ref].source != source for ref in refs):
                raise ValueError("evidence must refer only to the current source")
            if len(set(refs)) != len(refs):
                raise ValueError("evidence must not repeat a unit")
            evidence.append(tuple(refs))
        return cls(source, tuple(patches), tuple(evidence))

    def to_dict(self) -> dict:
        return {"source": self.source, "patches": [patch.to_dict() for patch in self.patches],
                "evidence": [list(refs) for refs in self.evidence]}


@dataclass(frozen=True)
class Frame:
    source: str
    texts: tuple[str, ...]
    evidence: tuple[tuple[str, ...], ...]

    def __post_init__(self):
        if len(self.texts) != len(self.evidence):
            raise ValueError("each reconstructed text unit needs evidence")

    def to_dict(self) -> dict:
        return {"source": self.source, "texts": list(self.texts),
                "evidence": [list(refs) for refs in self.evidence]}
