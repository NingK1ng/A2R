"""Source claims, conditions, and residual validation."""
from __future__ import annotations

from dataclasses import dataclass

from .units import QUERY, Unit


def _anchors(raw: object, units: dict[str, Unit], source: str) -> tuple[str, ...]:
    if source == QUERY or not isinstance(raw, list) or not raw:
        raise ValueError("anchors must be a nonempty list from the current candidate")
    if any(not isinstance(u, str) or u not in units or units[u].source != source for u in raw):
        raise ValueError(f"anchors must refer only to units of source {source!r}")
    if len(set(raw)) != len(raw):
        raise ValueError("duplicate unit anchors")
    return tuple(raw)


@dataclass(frozen=True)
class Condition:
    kind: str
    text: str


@dataclass(frozen=True)
class Claim:
    target: str  # Includes modality and quantifiers.
    polarity: str
    scope: str
    condition: Condition | None  # Applies to the signed target.
    text: str
    units: tuple[str, ...]

    @classmethod
    def parse(cls, raw: dict, units: dict[str, Unit], source: str) -> Claim:
        fields = {"target", "polarity", "scope", "condition", "text", "units"}
        if not isinstance(raw, dict) or set(raw) != fields:
            raise ValueError("claim must contain exactly target, polarity, scope, condition, text, units")
        for field in ("target", "text"):
            if not isinstance(raw[field], str) or not raw[field].strip():
                raise ValueError(f"claim {field} must be a nonempty string")
        if raw["polarity"] not in ("+", "-"):
            raise ValueError("claim polarity must be '+' or '-'")
        if not isinstance(raw["scope"], str):
            raise ValueError("claim scope must be a string; use an empty string when unstated")
        condition = raw["condition"]
        if condition is not None:
            if not isinstance(condition, dict) or set(condition) != {"kind", "text"}:
                raise ValueError("condition must be null or an object containing exactly kind and text")
            if condition["kind"] not in ("if", "only_if", "iff", "unless", "when", "complex"):
                raise ValueError("invalid condition kind")
            if not isinstance(condition["text"], str) or not condition["text"].strip():
                raise ValueError("condition text must be a nonempty string")
            condition = Condition(condition["kind"], condition["text"].strip())
        return cls(
            raw["target"].strip(), raw["polarity"], raw["scope"].strip(),
            condition, raw["text"].strip(), _anchors(raw["units"], units, source),
        )

    def to_dict(self) -> dict:
        return {
            "target": self.target,
            "polarity": self.polarity,
            "scope": self.scope,
            "condition": None if self.condition is None else {
                "kind": self.condition.kind, "text": self.condition.text,
            },
            "text": self.text,
            "units": list(self.units),
        }


@dataclass
class Frame:
    source: str
    claims: dict[str, Claim]

    def to_dict(self) -> dict:
        return {"source": self.source, "claims": {key: claim.to_dict() for key, claim in self.claims.items()}}


@dataclass
class Residual:
    source: str
    keep: dict[str, tuple[str, ...]]
    drop: tuple[str, ...]
    replace: dict[str, Claim]
    add: tuple[Claim, ...]

    @classmethod
    def parse(cls, raw: dict, prediction: dict[str, Claim], units: dict[str, Unit], source: str) -> Residual:
        """Validate edit coverage and current-source anchors."""
        if not isinstance(raw, dict) or set(raw) != {"keep", "drop", "replace", "add"}:
            raise ValueError("residual must contain exactly keep, drop, replace, add")
        if not isinstance(raw["keep"], dict) or not isinstance(raw["replace"], dict):
            raise ValueError("keep and replace must be objects keyed by prediction IDs")
        if not isinstance(raw["drop"], list) or not isinstance(raw["add"], list):
            raise ValueError("drop and add must be lists")
        ids = [*raw["keep"], *raw["drop"], *raw["replace"]]
        if any(not isinstance(key, str) for key in ids):
            raise ValueError("prediction IDs must be strings")
        if len(set(ids)) != len(ids):
            raise ValueError("keep, drop and replace must not repeat or overlap prediction IDs")
        if set(ids) != set(prediction):
            missing, unknown = set(prediction) - set(ids), set(ids) - set(prediction)
            raise ValueError(f"residual coverage mismatch: missing={sorted(missing)}, unknown={sorted(unknown)}")
        return cls(
            source,
            {key: _anchors(refs, units, source) for key, refs in raw["keep"].items()},
            tuple(raw["drop"]),
            {key: Claim.parse(claim, units, source) for key, claim in raw["replace"].items()},
            tuple(Claim.parse(claim, units, source) for claim in raw["add"]),
        )

    def to_dict(self) -> dict:
        return {
            "keep": {key: list(refs) for key, refs in self.keep.items()},
            "drop": list(self.drop),
            "replace": {key: claim.to_dict() for key, claim in self.replace.items()},
            "add": [claim.to_dict() for claim in self.add],
        }


@dataclass
class Reading:
    units: dict[str, Unit]
    frames: dict[str, Frame]

    def to_dict(self) -> dict:
        return {
            "units": [{"id": unit.id, "source": unit.source, "text": unit.text} for unit in self.units.values()],
            "frames": {source: frame.to_dict() for source, frame in self.frames.items()},
        }
