"""Closed-loop prediction, exact residuals, semantic quantization, and reconstruction."""
from __future__ import annotations

from collections.abc import Callable
from difflib import SequenceMatcher

from .alignment import encode_gap, select_alignment
from .prompts import semantic_alignments
from .records import Error, Frame, Packet
from .units import Unit

LLM = Callable[[str], str]


def predict(previous: Frame | None) -> tuple[str, ...]:
    """Both encoder and decoder predict from the same reconstructed history."""
    return () if previous is None else previous.texts


def difference(prediction: tuple[str, ...], current: tuple[Unit, ...]) -> Error:
    """Keep the exact unit edits before making any semantic judgment."""
    target = tuple(unit.text for unit in current)
    spans = SequenceMatcher(None, prediction, target, autojunk=False).get_opcodes()
    return Error(prediction, current, tuple(spans))


def quantize(llm: LLM, query: str, source: str, error: Error) -> Packet:
    """Reuse equivalent subranges; transmit every remaining difference literally."""
    candidates = semantic_alignments(llm, query, error)
    patches, evidence = [], []

    def emit_gap(old_start: int, old_stop: int, new_start: int, new_stop: int) -> None:
        patch, refs = encode_gap(error, old_start, old_stop, new_start, new_stop)
        if patch is not None:
            patches.append(patch)
        evidence.extend(refs)

    for block, (kind, old_start, old_stop, new_start, new_stop) in enumerate(error.spans):
        if kind == "equal":
            evidence.extend((unit.id,) for unit in error.current[new_start:new_stop])
            continue

        matches = select_alignment(error, block, candidates) if kind == "replace" else ()
        old_cursor, new_cursor = old_start, new_start
        for match in matches:
            emit_gap(old_cursor, match.old_start, new_cursor, match.new_start)
            refs = tuple(unit.id for unit in error.current[match.new_start:match.new_stop])
            evidence.extend([refs] * (match.old_stop - match.old_start))
            old_cursor, new_cursor = match.old_stop, match.new_stop
        emit_gap(old_cursor, old_stop, new_cursor, new_stop)

    return Packet(source, tuple(patches), tuple(evidence))


def reconstruct(prediction: tuple[str, ...], packet: Packet) -> Frame:
    """Apply ordered patches in original prediction coordinates, without a model."""
    texts, cursor = [], 0
    for patch in packet.patches:
        if not cursor <= patch.start <= patch.stop <= len(prediction):
            raise ValueError("patches must be ordered, nonoverlapping prediction intervals")
        texts.extend(prediction[cursor:patch.start])
        texts.extend(patch.text)
        cursor = patch.stop
    texts.extend(prediction[cursor:])
    return Frame(packet.source, tuple(texts), packet.evidence)
