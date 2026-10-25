"""Prompts for reading structural residuals."""
from __future__ import annotations

import json
from typing import TYPE_CHECKING

from .units import Unit

if TYPE_CHECKING:
    from .reading import Claim


INNOVATION = """Read only the current source as evidence for what that source says about the query.
The prediction is a structural hypothesis, not evidence or a preferred answer.
Identify its innovations directly from the current original text. Return edit operations,
not a complete current-source graph followed by a computed difference. Do not choose a winner.

Query context (not source evidence):
{query}

Current source {source}, numbered original units:
{units}

Predicted structure (all entries await confirmation from the current source):
{prediction}

Return only a JSON object with four fields:
  "keep": {{"predicted_id": ["current_unit_id", "another_current_unit_id"]}},
  "drop": ["predicted_id"],
  "replace": {{"predicted_id": complete_claim}},
  "add": [complete_claim]

Every predicted identifier must occur exactly once across keep, drop, and replace.
For an empty prediction, keep and replace are empty objects, drop is empty, and all
relevant assertions go in add. Use no identifiers outside the prediction as edit keys.

Operations:
- KEEP only when the current source supports the entire predicted meaning, including
  target, polarity, scope, condition, and the full meaning of text. Equivalent wording
  is allowed. Supply fresh
  current-source anchors; never retain a claim merely because the prediction contains it.
- DROP when that predicted assertion is not stated by the current source. Silence is
  not denial. A dropped assertion adds no negative stance to the current source.
- REPLACE when a corresponding assertion changes its target, polarity, scope, condition,
  or condition attachment. Supply the complete current claim and its current anchors.
  Do not return a fragment such as "different condition" or silently inherit old fields.
- ADD assertions absent from the prediction, including extra positions on an existing
  target. A source may make conflicting assertions: preserve both instead of reconciling
  them. Do not force genuinely new content into an existing predicted claim.

A complete claim has exactly these fields:
{{"target": "the proposition", "polarity": "+", "scope": "",
 "condition": null, "text": "the full assertion", "units": ["current_unit_id"]}}

Preserve the original language and these distinctions:
- target retains the subject, modality, quantities, and quantifier scope. Reuse a
  predicted target only when its meaning is equivalent; sharing a topic is insufficient.
  It must contain all assertion logic not represented by scope or condition, including
  causal links and linked choices; those meanings cannot appear only in text.
  When that meaning is unchanged, copy the predicted target string exactly, even if
  the current source paraphrases it or changes its attached condition or scope.
  polarity is "+" for asserting that target and "-" for explicitly denying it.
  Keep inner negation in the target when moving it would change modal or quantifier
  scope: "must not launch" is not simply the denial of "must launch".
- scope records stated time, population, and other applicability limits; use "" when
  none is stated. Keep each limit attached to the assertion it actually qualifies.
- condition is null when no condition is stated, not a claim of universal applicability.
  Otherwise use {{"kind": "if|only_if|iff|unless|when|complex", "text": "original qualifier wording"}}.
  Choose one kind. "if C, P" makes C sufficient for the signed proposition P;
  "P only if C" makes C necessary for that signed proposition, including when polarity
  is "-". Preserve necessity versus sufficiency and negation within the condition.
  Conditions describe contingencies, not facts that have already happened.
- Preserve compound conditions, causal links, and either/or statements with their
  original wording and attachment. Use complex for conditions whose logic would be
  distorted by a simpler kind; do not invent separate facts or select one alternative.
- text gives a complete, faithful statement, including qualifiers and linked clauses;
  it need not be a verbatim quotation. Original wording remains available through units.
  Do not put additional independent assertions in text without representing them in
  the structural fields or as separate claims.
  units must be a nonempty list of current-source unit identifiers supporting that
  whole assertion. Several assertions may share a unit; one assertion may span units.
  The query and the prediction cannot supply an anchor or establish a source assertion.

Read all current-source units once more for relevant content the prediction does not
contain. Capture that content through add or a complete replacement. Leave out only
material unrelated to the query. Return no explanations or final answer."""


def innovation_prompt(
    query: str, source: str, units: dict[str, Unit], prediction: dict[str, Claim]
) -> str:
    """Format current source units and the prediction."""
    predicted = {claim_id: claim.to_dict() for claim_id, claim in prediction.items()}
    for claim in predicted.values():
        claim.pop("units", None)
    return INNOVATION.format(
        query=query,
        source=json.dumps(source, ensure_ascii=False),
        units="\n".join(f"[{u.id}] {u.text}" for u in units.values() if u.source == source),
        prediction=json.dumps(predicted, ensure_ascii=False, indent=2),
    )
