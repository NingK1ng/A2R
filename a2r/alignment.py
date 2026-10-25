"""Choose an ordered semantic reuse chain within one exact replacement error."""
from __future__ import annotations

import json

from .records import Alignment, Error, Patch

Evidence = tuple[tuple[str, ...], ...]


def encode_gap(error: Error, old_start: int, old_stop: int,
               new_start: int, new_stop: int) -> tuple[Patch | None, Evidence]:
    current = error.current[new_start:new_stop]
    evidence = tuple((unit.id,) for unit in current)
    if old_start == old_stop and new_start == new_stop:
        return None, evidence
    return Patch(old_start, old_stop, tuple(unit.text for unit in current)), evidence


def _record_bytes(value: object) -> int:
    """Charge UTF-8 record bytes plus one separator; omit fixed container bytes."""
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")) + 1


def _gap_cost(error: Error, old_start: int, old_stop: int,
              new_start: int, new_stop: int) -> int:
    patch, evidence = encode_gap(error, old_start, old_stop, new_start, new_stop)
    return (0 if patch is None else _record_bytes(patch.to_dict())) + sum(
        _record_bytes(refs) for refs in evidence
    )


def _reuse_cost(error: Error, match: Alignment) -> int:
    refs = tuple(unit.id for unit in error.current[match.new_start:match.new_stop])
    return (match.old_stop - match.old_start) * _record_bytes(refs)


def select_alignment(error: Error, block: int,
                     candidates: tuple[Alignment, ...]) -> tuple[Alignment, ...]:
    """Minimize additive patch/evidence record cost over monotone reuse chains."""
    if type(block) is not int or not 0 <= block < len(error.spans):
        raise ValueError("alignment block must identify an existing opcode")
    kind, old_start, old_stop, new_start, new_stop = error.spans[block]
    if kind != "replace":
        raise ValueError("semantic alignment applies only to replacement blocks")
    unique = {
        (match.old_start, match.old_stop, match.new_start, match.new_stop): match
        for match in candidates if match.block == block and match.equivalent
    }
    matches = [unique[key] for key in sorted(unique)]
    best_costs, best_chains = [], []
    selected = ()  # Sending the whole current block is always a feasible path.
    selected_cost = _gap_cost(error, old_start, old_stop, new_start, new_stop)
    for index, match in enumerate(matches):
        reuse_cost = _reuse_cost(error, match)
        cost = _gap_cost(error, old_start, match.old_start, new_start, match.new_start) + reuse_cost
        chain = (match,)
        for previous_index, previous in enumerate(matches[:index]):
            # A chain must advance without overlap in both original sequences.
            if previous.old_stop > match.old_start or previous.new_stop > match.new_start:
                continue
            transition = _gap_cost(error, previous.old_stop, match.old_start,
                                   previous.new_stop, match.new_start)
            candidate_cost = best_costs[previous_index] + transition + reuse_cost
            if candidate_cost < cost:
                cost = candidate_cost
                chain = best_chains[previous_index] + (match,)
        best_costs.append(cost)
        best_chains.append(chain)
        total = cost + _gap_cost(error, match.old_stop, old_stop, match.new_stop, new_stop)
        if total < selected_cost:
            selected_cost, selected = total, chain
    return selected
