"""Pure SRL span and dependency-graph utilities."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


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


_FUNCTION_POS = frozenset({"ADP", "SCONJ", "CCONJ", "DET", "PART"})
_QUANTIFIERS = frozenset(
    {
        "all",
        "some",
        "more",
        "lot",
        "lots",
        "enough",
        "none",
        "any",
        "most",
        "less",
        "much",
    }
)


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
    """Return sentence-relative content-head indices for one SRL span.

    The root of the span's dependency structure is used first. If that root is a function
    word, the nearest content-bearing descendant inside the span is selected. Coordinated
    heads are retained as additional heads for non-predicate roles.
    """

    if start < 0 or end > len(token_ids) or start >= end:
        raise ValueError(
            f"Invalid token span [{start}, {end}) for {len(token_ids)} tokens"
        )

    global_to_local = {int(token_id): index for index, token_id in enumerate(token_ids)}
    span_indices = tuple(range(start, end))
    span_set = set(span_indices)

    roots = [
        index
        for index in span_indices
        if head_token_ids[index] is None
        or global_to_local.get(int(head_token_ids[index]), -1) not in span_set
        or global_to_local.get(int(head_token_ids[index]), -1) == index
    ]
    non_punct_roots = [index for index in roots if dependencies[index] != "punct"]
    root = (non_punct_roots or roots or [start])[0]

    children: dict[int, list[int]] = {index: [] for index in span_indices}
    for child in span_indices:
        head = head_token_ids[child]
        if head is None:
            continue
        parent = global_to_local.get(int(head))
        if parent in span_set and parent != child:
            children[parent].append(child)

    def nearest_content_descendant(index: int) -> int:
        queue = list(children.get(index, ()))
        visited: set[int] = set()
        while queue:
            candidate = queue.pop(0)
            if candidate in visited:
                continue
            visited.add(candidate)
            if (
                dependencies[candidate] != "punct"
                and pos[candidate] not in _FUNCTION_POS
            ):
                return candidate
            queue.extend(children.get(candidate, ()))
        return index

    if pos[root] in _FUNCTION_POS:
        root = nearest_content_descendant(root)

    if text[root].lower() in _QUANTIFIERS:
        of_children = [
            child for child in children.get(root, ()) if text[child].lower() == "of"
        ]
        if of_children:
            replacement = nearest_content_descendant(of_children[0])
            if replacement != of_children[0] or pos[replacement] not in _FUNCTION_POS:
                root = replacement

    heads = [root]
    if role != "V":
        # Retain every coordinated item whose dependency ancestry remains inside the span
        # and ultimately reaches the selected root or another retained conjunction.
        changed = True
        while changed:
            changed = False
            for index in span_indices:
                if index in heads or dependencies[index] != "conj":
                    continue
                parent_global = head_token_ids[index]
                parent = (
                    None
                    if parent_global is None
                    else global_to_local.get(int(parent_global))
                )
                if parent in heads:
                    heads.append(index)
                    changed = True

    return tuple(dict.fromkeys(heads))
