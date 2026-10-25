"""Closed-loop text residual coding with source attribution."""
from __future__ import annotations

import json
from dataclasses import dataclass

from .coding import LLM, difference, predict, quantize, reconstruct
from .records import Frame, Packet
from .units import Unit, number


@dataclass
class Result:
    units: dict[str, Unit]
    frames: dict[str, Frame]
    packets: dict[str, Packet]
    calls: int
    usage: dict | None = None

    @property
    def answer(self) -> str:
        groups: dict[tuple[str, ...], list[Frame]] = {}
        for frame in self.frames.values():
            groups.setdefault(frame.texts, []).append(frame)
        paragraphs = []
        for texts, frames in groups.items():
            lines = [", ".join(frame.source for frame in frames) + ":"]
            for i, text in enumerate(texts):
                refs = "; ".join(f"{f.source}: {', '.join(f.evidence[i])}" for f in frames)
                lines.append(f"{text} [{refs}]")
            paragraphs.append("\n".join(lines) if texts else lines[0] + " (empty)")
        return "\n\n".join(paragraphs)

    @property
    def coded_bytes(self) -> int:
        wire = [packet.to_dict() for packet in self.packets.values()]
        return len(json.dumps(wire, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))

    def to_dict(self) -> dict:
        return {"answer": self.answer,
                "units": [{"id": u.id, "source": u.source, "text": u.text} for u in self.units.values()],
                "frames": {s: frame.to_dict() for s, frame in self.frames.items()},
                "packets": {s: packet.to_dict() for s, packet in self.packets.items()},
                "calls": self.calls, "usage": self.usage, "coded_bytes": self.coded_bytes}


class DPCM:
    def __init__(self, llm: LLM):
        self.llm = llm

    def __call__(self, query: str, candidates: dict[str, str]) -> Result:
        if not candidates:
            raise ValueError("at least one source is required")
        start = dict(getattr(self.llm, "usage", {}))
        units = number(query, candidates)
        previous, frames, packets, calls = None, {}, {}, 0
        for source in candidates:
            current = tuple(u for u in units.values() if u.source == source)
            prediction = predict(previous)
            error = difference(prediction, current)
            packet = quantize(self.llm, query, source, error)
            previous = reconstruct(prediction, packet)
            frames[source], packets[source] = previous, packet
            calls += any(tag == "replace" for tag, *_ in error.spans)
        used = getattr(self.llm, "usage", None)
        usage = None if used is None else {k: v - start.get(k, 0) for k, v in used.items()}
        return Result(units, frames, packets, calls, usage)
