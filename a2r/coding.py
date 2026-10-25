"""Predictive reading of source claims."""
from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import replace
from itertools import combinations

from .prompts import innovation_prompt
from .reading import Claim, Frame, Reading, Residual
from .units import Unit

LLM = Callable[[str], str]


def predict(reference: Frame | None) -> dict[str, Claim]:
    """Copy the reference and clear its source anchors."""
    return {} if reference is None else {
        key: replace(claim, units=()) for key, claim in reference.claims.items()
    }


def innovation(llm: LLM, query: str, source: str, units: dict[str, Unit],
               prediction: dict[str, Claim]) -> Residual:
    """Read current-source edits against the prediction."""
    reply = llm(innovation_prompt(query, source, units, prediction))
    reply = re.sub(r"<think>.*?(</think>|$)", "", reply, flags=re.S)
    raw = json.loads(reply[reply.find("{"):reply.rfind("}") + 1])
    return Residual.parse(raw, prediction, units, source)


def reconstruct(prediction: dict[str, Claim], residual: Residual) -> Frame:
    """Apply edits and bind claims to current-source anchors."""
    claims = {
        key: replace(prediction[key], units=residual.keep[key])
        if key in residual.keep else residual.replace[key]
        for key in prediction if key not in residual.drop
    }
    claims.update({f"{residual.source}:{n}": claim
                   for n, claim in enumerate(residual.add, 1)})
    return Frame(residual.source, claims)


def reconcile(reading: Reading) -> tuple[str, list[dict]]:
    """Compare reconstructed claims and preserve their attribution."""
    targets: dict[str, list[tuple[str, str, Claim]]] = {}
    for source, frame in reading.frames.items():
        for key, claim in frame.claims.items():
            targets.setdefault(claim.target, []).append((source, key, claim))

    paragraphs, relations = [], []
    fields = ("polarity", "scope", "condition")
    for target, entries in targets.items():
        variants: dict[tuple, list[tuple[str, str, Claim]]] = {}
        for source, key, claim in entries:
            signature = (*[getattr(claim, f) for f in fields], claim.text)
            variants.setdefault(signature, []).append((source, key, claim))
        lines = [f"Regarding {json.dumps(target, ensure_ascii=False)}:"]
        for group in variants.values():
            sources = list(dict.fromkeys(source for source, _, _ in group))
            refs = list(dict.fromkeys(u for _, _, claim in group for u in claim.units))
            names = ", ".join(json.dumps(s, ensure_ascii=False) for s in sources)
            label = "Candidate" if len(sources) == 1 else "Candidates"
            lines.append(f"{label} {names}: {group[0][2].text} [{', '.join(refs)}]")
        paragraphs.append("\n".join(lines))

        for (s, a, x), (t, b, y) in combinations(entries, 2):
            if s == t:
                continue
            changes = [f for f in fields if getattr(x, f) != getattr(y, f)]
            relation = "equivalent" if not changes else (
                "conflict" if changes == ["polarity"] and x.condition is None
                else "qualified_difference"
            )
            relations.append({"sources": [s, t], "claims": [a, b],
                              "units": [list(x.units), list(y.units)],
                              "relation": relation, "changes": changes})

    for source, frame in reading.frames.items():
        if not frame.claims:
            paragraphs.append(f"Candidate {json.dumps(source, ensure_ascii=False)}: no relevant statements recorded.")
    return "\n\n".join(paragraphs), relations
