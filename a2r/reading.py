"""A reading of the candidates: the claims their units put forward on the query, which unit counts
for or against which claim and under what condition, and which claims are alternatives.

Relations between units are computed from the reading rather than read separately, so the relations
of a reading are always jointly consistent, and two readings of one case can be compared unit pair by
unit pair however their claims are worded.
"""
from __future__ import annotations

import re
from collections.abc import Collection
from dataclasses import dataclass
from itertools import combinations

from .units import QUERY, Unit

_CONDITIONAL = re.compile(r"^(?:only if|if|when|whenever|unless|provided)\b", re.I)


def _plain(text: str) -> str:
    """Normalised wording, to recognise a condition restated as a claim."""
    text = re.sub(r"\s+", " ", text.strip().rstrip(".!?;:").lower())
    return re.sub(r"^(?:only if|if|when|whenever|provided(?: that)?) ", "", text)


@dataclass(frozen=True)
class Stance:
    unit: str
    claim: str
    polarity: str  # "+": the unit counts for the claim, "-": against it
    condition: str | None  # the stance holds only under this circumstance


@dataclass
class Reading:
    units: dict[str, Unit]
    claims: dict[str, str]  # id -> source-neutral clause, in the order an answer presents them
    stances: list[Stance]
    issues: list[tuple[str, ...]]  # claims grouped into mutually exclusive alternatives; singletons included

    @classmethod
    def parse(cls, raw: dict, units: dict[str, Unit]) -> Reading:
        """Validate a reading proposed by a model. Claims worded alike are taken as one claim; every
        other problem is reported in one ValueError, so that a single repair can address all of them."""
        problems: list[str] = []
        claims: dict[str, str] = {}
        same: dict[str, str] = {}  # a claim id whose text repeats an earlier claim -> the earlier id
        for c in raw.get("claims") or []:
            text = (c.get("text") or "").strip()
            if c["id"] in claims or c["id"] in same:
                problems.append(f"duplicate claim id {c['id']!r}")
            elif not text:
                problems.append(f"claim {c['id']!r} has no text")
            elif any(_plain(text) == _plain(t) for t in claims.values()):
                same[c["id"]] = next(k for k, t in claims.items() if _plain(text) == _plain(t))
            else:
                claims[c["id"]] = text
                if _CONDITIONAL.match(text):
                    problems.append(
                        f"claim {c['id']!r} ({text!r}) contains a condition: keep as the claim only what holds under "
                        f"the condition, and give the condition as the if of the stances that take it"
                    )
        if not claims:
            raise ValueError("no claims")

        stances: list[Stance] = []
        for s in raw.get("stances") or []:
            condition = (s.get("if") or "").strip()
            claim = same.get(s["claim"], s["claim"])
            stance = Stance(s["unit"], claim, s["polarity"], None if condition.lower() in ("", "none", "null") else condition)
            if stance.unit not in units:
                problems.append(f"stance of unknown unit {stance.unit!r}")
            elif stance.claim not in claims:
                problems.append(f"stance on unknown claim {stance.claim!r}")
            elif stance.polarity not in ("+", "-"):
                problems.append(f"polarity must be '+' or '-', got {stance.polarity!r}")
            elif stance not in stances:
                stances.append(stance)
        for a, b in combinations(stances, 2):
            if (a.unit, a.claim, a.condition) == (b.unit, b.claim, b.condition):
                problems.append(f"{a.unit} counts both for and against {a.claim} under the same condition")
        own = {(s.unit, _plain(s.condition)) for s in stances if s.condition}
        for s in stances:
            if s.condition is None and (s.unit, _plain(claims[s.claim])) in own:
                problems.append(
                    f"{s.unit} takes a stance on {s.claim} ({claims[s.claim]!r}) and makes another of its stances hold "
                    f"only under it. If {s.unit} merely names it as a condition, drop the stance of {s.unit} on {s.claim} "
                    f"(and {s.claim} itself if no other stance is left on it). If {s.unit} states it, its other stance does "
                    f"not depend on it: make that stance unconditional, or drop it. Leave all other stances as they are"
                )
        for c in claims:
            if not any(s.claim == c for s in stances):
                problems.append(f"claim {c!r} has no stance: drop it, or attach the units that make it")

        root = {c: c for c in claims}

        def find(c: str) -> str:
            while root[c] != c:
                c = root[c]
            return c

        for group in raw.get("alternatives") or []:
            group = [same.get(c, c) for c in group]
            problems += [f"unknown claim {c!r} among alternatives" for c in group if c not in claims]
            known = [c for c in group if c in claims]
            for c in known[1:]:
                root[find(c)] = find(known[0])
        if problems:
            raise ValueError("; ".join(problems))
        issues: dict[str, list[str]] = {}
        for c in claims:
            issues.setdefault(find(c), []).append(c)
        return cls(units, claims, stances, [tuple(issue) for issue in issues.values()])

    def issue_of(self, claim: str) -> tuple[str, ...]:
        return next(issue for issue in self.issues if claim in issue)

    def stances_of(self, units: Collection[str]) -> frozenset[tuple[str, str, str | None]]:
        return frozenset((s.claim, s.polarity, s.condition) for s in self.stances if s.unit in units)

    def sources(self, units: Collection[str]) -> list[str]:
        """Sources of the units: candidates in order, the query last."""
        found = {self.units[u].source for u in units}
        candidates = [s for s in dict.fromkeys(u.source for u in self.units.values()) if s != QUERY]
        return [s for s in candidates if s in found] + ([QUERY] if QUERY in found else [])

    def relate(self, xs: Collection[str], ys: Collection[str]) -> tuple[str, str]:
        """(relation, operation) of two groups of units.

        Units without stances bear on nothing under the query. Units conflict if one counts against
        a claim the other counts for, or both support different alternatives; the conflict is
        conditional if every such clash involves a stance taken only under a condition. Otherwise
        units with the same stances are equivalent, units whose stances include the other's subsume
        them, and the rest add to each other, conditionally if a claim they share carries a condition.
        """
        a, b = self.stances_of(xs), self.stances_of(ys)
        if not a or not b:
            return "query_invalid", "drop"
        clashes = [(s, t) for s in a for t in b if self._clash(s, t)]
        if clashes:
            return "conflict", "condition" if all(s[2] or t[2] for s, t in clashes) else "report_unresolved"
        if a == b:
            return "equivalent", "merge"
        shared = {c for c, _, _ in a} & {c for c, _, _ in b}
        operation = "condition" if any(g for c, _, g in a | b if c in shared) else "merge"
        return ("subsumes" if a < b or b < a else "additive"), operation

    def _clash(self, s: tuple, t: tuple) -> bool:
        if s[0] == t[0]:
            return s[1] != t[1]
        return s[1] == t[1] == "+" and t[0] in self.issue_of(s[0])

    def relation(self, x: str, y: str) -> tuple[str, str]:
        return self.relate([x], [y])

    def relations(self) -> list[dict]:
        """Pairs of units from different sources whose relation is not additive/merge, the subsuming
        unit first. Unlisted pairs are additive/merge, or query_invalid/drop if a unit has no stance."""
        found = []
        for x, y in combinations(self.units, 2):
            if self.units[x].source == self.units[y].source or not self.stances_of([x]) or not self.stances_of([y]):
                continue
            relation, operation = self.relation(x, y)
            if relation == "subsumes" and self.stances_of([x]) < self.stances_of([y]):
                x, y = y, x
            if (relation, operation) != ("additive", "merge"):
                found.append({"units": [x, y], "relation": relation, "operation": operation})
        return found

    def signature(self) -> dict[tuple[str, str], tuple[str, str]]:
        """The relation of every pair of units."""
        return {(x, y): self.relation(x, y) for x, y in combinations(self.units, 2)}

    def to_dict(self) -> dict:
        return {
            "units": [{"id": u.id, "source": u.source, "text": u.text} for u in self.units.values()],
            "claims": [
                {
                    "id": c,
                    "text": text,
                    "stances": [{"unit": s.unit, "polarity": s.polarity, "if": s.condition} for s in self.stances if s.claim == c],
                }
                for c, text in self.claims.items()
            ],
            "alternatives": [list(issue) for issue in self.issues if len(issue) > 1],
            "irrelevant": [u for u, unit in self.units.items() if unit.source != QUERY and not self.stances_of([u])],
            "relations": self.relations(),
        }
