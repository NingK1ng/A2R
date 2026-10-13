"""The reading prompt, the only one the method uses."""
from __future__ import annotations

from .units import QUERY, Unit

READ = """Read the candidate answers below as an arbiter must before combining them: record what each numbered unit says about the query, without judging who is right.

{units}

Return the reading in three fields.

"claims": the distinct propositions that the units put forward on this query, in the order an answer should present them, beginning with those that answer the query. Write each as a source-neutral clause that could follow "that": keep the candidates' own wording and numbers, and leave out who says it and any condition under which it holds. Units of different candidates that say the same thing share one claim. Every claim must be taken up by some stance; add no claim that no unit makes.

"stances": for every unit that bears on the query, the claims it counts for ("+") or against ("-"). A unit counts for the claims it states, caveats and limitations included, and against the claims it disputes; a fact also counts for or against the conclusion its candidate draws from it. If the unit takes a stance only under some circumstance (a condition, threshold, scope or prerequisite), give that circumstance in "if" as a plain clause and the polarity the unit takes under it; otherwise "if" is null. For example, "If costs exceed the budget, do not hire" counts against "the clinic should hire a second nurse" with "if": "costs exceed the budget", and makes no claim that costs exceed the budget. A unit that bears on nothing under this query has no stance, and units of the query take stances only on what the query itself asserts.

"alternatives": groups of rival answers to the query: courses of action or verdicts of which only one can be followed or be true (e.g. renting the space and buying it). Facts, evidence and caveats are not alternatives.

Relations depend on the query: two units can be alternatives under one query and compatible under another.

Return only JSON:
{{"claims": [{{"id": "c1", "text": "..."}}],
 "stances": [{{"unit": "A1", "claim": "c1", "polarity": "+", "if": null}}],
 "alternatives": [["c1", "c2"]]}}"""


def read(units: dict[str, Unit], order: list[str]) -> str:
    """The units grouped by source: the query first, then the candidates in the given order."""
    blocks: dict[str, list[str]] = {}
    for u in units.values():
        blocks.setdefault(u.source, []).append(f"[{u.id}] {u.text}")
    listed = "\n\n".join(
        ("Query:" if s == QUERY else f"Candidate {s}:") + "\n" + "\n".join(blocks[s]) for s in [QUERY, *order] if s in blocks
    )
    return READ.format(units=listed)
