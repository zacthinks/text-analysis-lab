"""Pure SRL span and dependency-graph utilities."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from text_analysis_lab.linguistics.heads import resolve_semantic_head_indices


class InvalidBioSequence(ValueError):
    """Raised when a tag sequence cannot represent valid BIO spans."""

    def __init__(self, position: int, tag: str, message: str) -> None:
        self.position = position
        self.tag = tag
        super().__init__(f"Invalid BIO tag at position {position}: {tag!r}. {message}")


@dataclass(frozen=True, slots=True)
class BioSpan:
    label: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class BioRepair:
    """One deterministic repair applied after WordPiece-to-token projection."""

    position: int
    original_tag: str
    repaired_tag: str
    reason: str


def repair_projected_bio_tags(
    tags: Sequence[str],
) -> tuple[tuple[str, ...], tuple[BioRepair, ...]]:
    """Repair token-level BIO transitions created by WordPiece projection.

    The converted AllenNLP model is decoded with valid BIO constraints over WordPieces.
    Selecting one prediction per original token can nevertheless expose an ``I-LABEL``
    whose supporting ``B-LABEL`` occurred on a non-selected continuation WordPiece.  At
    token level that tag must begin a new span, so only orphaned or label-mismatched
    ``I-LABEL`` tags are promoted deterministically to ``B-LABEL``.  Valid transitions
    and all model labels remain unchanged.
    """

    repaired: list[str] = []
    repairs: list[BioRepair] = []
    active_label: str | None = None

    for position, raw_tag in enumerate(tags):
        tag = str(raw_tag)
        if tag == "O":
            repaired.append(tag)
            active_label = None
            continue
        if "-" not in tag:
            raise InvalidBioSequence(position, tag, "Expected O, B-LABEL, or I-LABEL.")
        prefix, label = tag.split("-", 1)
        if not label or prefix not in {"B", "I"}:
            raise InvalidBioSequence(position, tag, "Expected O, B-LABEL, or I-LABEL.")
        if prefix == "B":
            repaired.append(tag)
            active_label = label
            continue
        if active_label == label:
            repaired.append(tag)
            continue

        replacement = f"B-{label}"
        reason = "orphan_inside" if active_label is None else "mismatched_inside"
        repaired.append(replacement)
        repairs.append(
            BioRepair(
                position=position,
                original_tag=tag,
                repaired_tag=replacement,
                reason=reason,
            )
        )
        active_label = label

    return tuple(repaired), tuple(repairs)


def bio_spans(tags: Sequence[str]) -> tuple[BioSpan, ...]:
    """Convert BIO tags to half-open spans, rejecting malformed sequences."""

    spans: list[BioSpan] = []
    active_label: str | None = None
    active_start = 0

    for index, tag in enumerate((*tags, "O")):
        if tag == "O":
            if active_label is not None:
                spans.append(BioSpan(active_label, active_start, index))
                active_label = None
            continue
        if "-" not in tag:
            raise InvalidBioSequence(index, tag, "Expected O, B-LABEL, or I-LABEL.")
        prefix, label = tag.split("-", 1)
        if not label or prefix not in {"B", "I"}:
            raise InvalidBioSequence(index, tag, "Expected O, B-LABEL, or I-LABEL.")
        if prefix == "B":
            if active_label is not None:
                spans.append(BioSpan(active_label, active_start, index))
            active_label = label
            active_start = index
            continue
        if active_label != label:
            raise InvalidBioSequence(
                index,
                tag,
                "I-LABEL must follow B-LABEL or I-LABEL with the same label.",
            )

    return tuple(spans)


def content_head_indices(
    *,
    start: int,
    end: int,
    token_ids: Sequence[int],
    head_token_ids: Sequence[int | None],
    dependencies: Sequence[str | None],
    pos: Sequence[str | None],
    text: Sequence[str],
    role: str,
) -> tuple[int, ...]:
    """Backward-compatible wrapper around configurable semantic-head resolution."""

    return resolve_semantic_head_indices(
        start=start,
        end=end,
        token_ids=token_ids,
        head_token_ids=head_token_ids,
        dependencies=dependencies,
        pos=pos,
        text=text,
        role=role,
    )
