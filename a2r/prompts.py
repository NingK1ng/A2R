"""Read semantic correspondences inside exact text-replacement blocks."""
from __future__ import annotations

import json
import re
from collections.abc import Callable

from .records import Alignment, Error


QUANTIZATION = """Find candidate semantic correspondences inside the eligible
replacement blocks below. The encoder may reuse an old span only when BOTH entailment
directions are true. It will transmit all uncovered current text exactly. Return
coordinates and boolean judgments only; do not generate replacement text.

Query context:
{query}

Full prediction, with zero-based unit indices:
{prediction}

Full current source, with zero-based indices and original evidence IDs:
{current}

Eligible replacement blocks:
{blocks}

Return only JSON:
{{"alignments": [{{"block": 0, "old": [0, 1], "new": [0, 2],
                  "old_to_new": true, "new_to_old": true}}]}}

"block" is an original replacement opcode ID from the eligible list.
"old" and "new" are GLOBAL half-open index ranges [start, stop), not offsets inside
the block. Each range must be nonempty, continuous, and wholly inside its listed
replacement block. Use the supplied whole-unit boundaries; never select word fragments.
One old unit may correspond to several new units, and several old units to one new unit.

A replacement block may mix paraphrases with changed assertions. Look for smaller
complete correspondences inside such blocks rather than judging only the entire block.
Propose useful alternatives even when their ranges overlap; the encoder will choose
an ordered, nonoverlapping subset. Do not repeat the same block/old/new correspondence.
You need not cover every unit or block. An empty alignments list means no candidates;
uncovered current text is retained, never discarded.

old_to_new asks whether the COMPLETE old span entails the COMPLETE new span.
new_to_old asks whether the COMPLETE new span entails the COMPLETE old span.
Report each direction separately; a one-way correspondence may be included but will
not be reused. Use false for an uncertain direction. True/true requires interchangeable
complete meanings in their full contexts, not shared topics, similar words, partial
overlap, or an inference based on outside facts.

Preserve negation and its scope, modality, quantities, quantifiers, attribution, time,
population, necessary versus sufficient conditions, causal links, and linked choices.
Do not detach a condition, exception, or referent from the assertion it qualifies.
"Release only if checks pass" is not interchangeable with "release" or "if checks pass,
release". Do not ignore a meaning difference because it seems unimportant to the query.
The prediction is not evidence or a preferred answer. Preserve contradictory assertions.

Check context jointly: for ANY subset of proposed true/true correspondences that can
be selected together without overlapping or crossing in either sequence, substituting
those old spans must preserve the complete current-source meaning. This includes
cross-span conditions and pronoun references. If equivalence depends on also replacing
another span, expand to a complete span that contains the dependency, or do not mark
both directions true. Never assume that the encoder will select another candidate.
Insertions, deletions, and equal blocks are ineligible.
Return no explanations, confidence scores, generated text, or additional fields."""


def semantic_alignments(llm: Callable[[str], str], query: str,
                        error: Error) -> tuple[Alignment, ...]:
    blocks = {
        index: (a, b, c, d)
        for index, (tag, a, b, c, d) in enumerate(error.spans)
        if tag == "replace" and a < b and c < d
    }
    if not blocks:
        return ()
    prompt = QUANTIZATION.format(
        query=query,
        prediction=json.dumps([{"index": i, "text": text}
                               for i, text in enumerate(error.prediction)], ensure_ascii=False),
        current=json.dumps([{"index": i, "id": unit.id, "text": unit.text}
                            for i, unit in enumerate(error.current)], ensure_ascii=False),
        blocks=json.dumps([{"block": index, "old": [a, b], "new": [c, d]}
                           for index, (a, b, c, d) in blocks.items()], ensure_ascii=False),
    )
    reply = re.sub(r"<think>.*?(</think>|$)", "", llm(prompt), flags=re.S).strip()
    fence = "```"
    reply = re.sub(rf"^{fence}(?:json)?\s*|\s*{fence}$", "", reply, flags=re.I).strip()

    def unique_object(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON field: {key}")
            value[key] = item
        return value

    raw = json.loads(reply, object_pairs_hook=unique_object)
    if not isinstance(raw, dict) or set(raw) != {"alignments"} or not isinstance(raw["alignments"], list):
        raise ValueError("semantic reader must return exactly an alignments list")
    alignments, seen = [], set()
    fields = {"block", "old", "new", "old_to_new", "new_to_old"}
    for item in raw["alignments"]:
        if not isinstance(item, dict) or set(item) != fields:
            raise ValueError("each alignment needs exactly block, old, new, old_to_new, new_to_old")
        block = item["block"]
        if type(block) is not int or block not in blocks:
            raise ValueError("alignment block must be an eligible replacement opcode")
        old, new = item["old"], item["new"]
        if any(not isinstance(span, list) or len(span) != 2
               or any(type(index) is not int for index in span) for span in (old, new)):
            raise ValueError("alignment ranges must be pairs of integer unit coordinates")
        a, b, c, d = blocks[block]
        if not (a <= old[0] < old[1] <= b and c <= new[0] < new[1] <= d):
            raise ValueError("alignment ranges must be nonempty and inside their replacement block")
        if type(item["old_to_new"]) is not bool or type(item["new_to_old"]) is not bool:
            raise ValueError("entailment judgments must be booleans")
        key = (block, *old, *new)
        if key in seen:
            raise ValueError("alignment correspondences must be unique")
        seen.add(key)
        alignments.append(Alignment(block, old[0], old[1], new[0], new[1],
                                    item["old_to_new"], item["new_to_old"]))
    return tuple(alignments)
