"""Arbitration: read the candidates once for every order in which they can be presented, keep the
reading that disagrees least with the others, and write it out."""
from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass

from . import prompts
from .reading import Reading
from .units import number
from .writing import write

LLM = Callable[[str], str]


def parse_json(text: str) -> dict:
    text = re.sub(r"<think>.*?(</think>|$)", "", text, flags=re.S)  # reasoning traces of R1-style models
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    if fenced:
        return json.loads(fenced[-1])
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("no JSON object in the reply")
    return json.loads(text[start : end + 1])


def ask_json(ask: LLM, prompt: str, validate: Callable[[dict], object] = lambda raw: raw):
    """One repair round: an unusable reply is sent back once with the error, to be edited rather than redone."""
    reply = ask(prompt)
    try:
        return validate(parse_json(reply))
    except (ValueError, KeyError, TypeError, AttributeError) as error:
        retry = (
            f"{prompt}\n\nYour previous reply:\n{reply.strip()}\n\nIt is unusable: {error!r}. "
            "Fix exactly these problems, keep everything else unchanged, and reply with the corrected JSON only."
        )
        return validate(parse_json(ask(retry)))


def medoid(readings: list[Reading]) -> tuple[int, list[int]]:
    """Index of the reading with the least total distance to the others, and every reading's
    distance to it. The distance between two readings is the number of unit pairs they relate
    differently. Ties go to the earlier reading."""
    signatures = [r.signature() for r in readings]

    def distance(a: dict, b: dict) -> int:
        return sum(a[pair] != b[pair] for pair in a)

    totals = [sum(distance(s, other) for other in signatures) for s in signatures]
    best = totals.index(min(totals))
    return best, [distance(signatures[best], s) for s in signatures]


@dataclass
class Arbitration:
    answer: str
    reading: Reading  # the selected reading; the answer writes it out
    looks: list[dict]  # per presentation: the order, then the relations read and their distance to the selection, or the error
    calls: int
    usage: dict | None = None  # tokens spent on this case, if the LLM client counts them

    def to_dict(self) -> dict:
        return {"answer": self.answer, "reading": self.reading.to_dict(), "looks": self.looks, "calls": self.calls, "usage": self.usage}


class Arbiter:
    def __init__(self, llm: LLM, rounds: int = 1):
        self.llm = llm
        self.rounds = rounds  # passes over all orders; repeated passes differ only if the model samples

    def read(self, query: str, candidates: dict[str, str], ask: LLM | None = None) -> tuple[Reading, list[dict]]:
        """Read the candidates in every cyclic order of their labels, `rounds` times over, and select
        the medoid reading. A reading that is still unusable after one repair is left out."""
        ask = ask or self.llm
        labels = sorted(candidates)
        units = number(query, {label: candidates[label] for label in labels})
        looks, readings = [], []
        total = self.rounds * len(labels)
        for n in range(total):
            look = {"order": labels[n % len(labels) :] + labels[: n % len(labels)]}
            looks.append(look)
            try:
                readings.append((look, ask_json(ask, prompts.read(units, look["order"]), lambda raw: Reading.parse(raw, units))))
            except (ValueError, KeyError, TypeError, AttributeError) as error:
                if not readings and n == total - 1:
                    raise
                look["error"] = repr(error)
        best, distances = medoid([r for _, r in readings])
        for (look, r), d in zip(readings, distances):
            look.update(relations=r.relations(), distance=d)
        readings[best][0]["selected"] = True
        return readings[best][1], looks

    def __call__(self, query: str, candidates: dict[str, str]) -> Arbitration:
        calls = 0
        start = dict(getattr(self.llm, "usage", {}))

        def ask(prompt: str) -> str:
            nonlocal calls
            calls += 1
            return self.llm(prompt)

        reading, looks = self.read(query, candidates, ask)
        used = getattr(self.llm, "usage", None)
        usage = None if used is None else {k: v - start.get(k, 0) for k, v in used.items()}
        return Arbitration(write(reading), reading, looks, calls, usage)

    def relate(self, query: str, u: str, v: str) -> str:
        """Query-conditioned relation of two texts: read them as candidates A and B and relate their units."""
        reading, _ = self.read(query, {"A": u, "B": v})
        side = {s: [i for i, unit in reading.units.items() if unit.source == s] for s in ("A", "B")}
        return reading.relate(side["A"], side["B"])[0]
