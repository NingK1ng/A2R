"""Use the first decoded source as a fixed predictor."""
from __future__ import annotations

from dataclasses import dataclass

from .coding import LLM, innovation, predict, reconstruct, reconcile
from .reading import Reading, Residual
from .units import number


@dataclass
class Arbitration:
    answer: str
    reading: Reading
    reference: str
    residuals: dict[str, Residual]
    relations: list[dict]
    calls: int
    usage: dict | None = None

    def to_dict(self) -> dict:
        return {"answer": self.answer, "reading": self.reading.to_dict(),
                "reference": self.reference,
                "residuals": {s: e.to_dict() for s, e in self.residuals.items()},
                "relations": self.relations, "calls": self.calls, "usage": self.usage}


class Arbiter:
    def __init__(self, llm: LLM):
        self.llm = llm

    def __call__(self, query: str, candidates: dict[str, str]) -> Arbitration:
        if not candidates:
            raise ValueError("at least one candidate is required")
        start = dict(getattr(self.llm, "usage", {}))
        units = number(query, candidates)
        reference, frames, residuals = None, {}, {}
        for source in sorted(candidates):
            prediction = predict(reference)
            residuals[source] = innovation(self.llm, query, source, units, prediction)
            frames[source] = reconstruct(prediction, residuals[source])
            if reference is None:
                reference = frames[source]
        reading = Reading(units, frames)
        answer, relations = reconcile(reading)
        used = getattr(self.llm, "usage", None)
        usage = None if used is None else {key: value - start.get(key, 0) for key, value in used.items()}
        return Arbitration(answer, reading, reference.source, residuals, relations, len(frames), usage)
