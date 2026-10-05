"""Configurable semantic-head resolution over dependency-aligned token spans."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


_DEFAULT_FUNCTION_POS = ("ADP", "SCONJ", "CCONJ", "DET", "PART")
_DEFAULT_QUANTIFIERS = (
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
)


@dataclass(frozen=True, slots=True)
class SemanticHeadRules:
    """Serializable policy controlling cheap semantic-head selection.

    Defaults reproduce the pre-policy content_head_indices behavior: move off common
    function POS tags, rewrite quantifier-plus-of constructions to their content
    object, and retain coordinated heads except for predicate roles.
    """

    function_pos: tuple[str, ...] = _DEFAULT_FUNCTION_POS
    move_off_function_pos: bool = True
    quantifiers: tuple[str, ...] = _DEFAULT_QUANTIFIERS
    quantifier_links: tuple[str, ...] = ("of",)
    rewrite_quantifier_links: bool = True
    preserve_coordination: bool = True
    coordination_excluded_roles: tuple[str, ...] = ("V",)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "function_pos",
            tuple(dict.fromkeys(str(value).upper() for value in self.function_pos)),
        )
        object.__setattr__(
            self,
            "quantifiers",
            tuple(dict.fromkeys(str(value).casefold() for value in self.quantifiers)),
        )
        object.__setattr__(
            self,
            "quantifier_links",
            tuple(
                dict.fromkeys(str(value).casefold() for value in self.quantifier_links)
            ),
        )
        object.__setattr__(
            self,
            "coordination_excluded_roles",
            tuple(
                dict.fromkeys(
                    str(value).upper() for value in self.coordination_excluded_roles
                )
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return stable JSON-safe state."""

        return {
            "function_pos": list(self.function_pos),
            "move_off_function_pos": self.move_off_function_pos,
            "quantifiers": list(self.quantifiers),
            "quantifier_links": list(self.quantifier_links),
            "rewrite_quantifier_links": self.rewrite_quantifier_links,
            "preserve_coordination": self.preserve_coordination,
            "coordination_excluded_roles": list(self.coordination_excluded_roles),
        }

    @classmethod
    def from_dict(cls, state: Mapping[str, Any]) -> "SemanticHeadRules":
        """Restore a policy from to_dict state."""

        values = dict(state)
        for name in (
            "function_pos",
            "quantifiers",
            "quantifier_links",
            "coordination_excluded_roles",
        ):
            if name in values:
                values[name] = tuple(values[name])
        return cls(**values)


def resolve_semantic_head_indices(
    *,
    start: int,
    end: int,
    token_ids: Sequence[int],
    head_token_ids: Sequence[int | None],
    dependencies: Sequence[str | None],
    pos: Sequence[str | None],
    text: Sequence[str],
    role: str,
    rules: SemanticHeadRules | None = None,
) -> tuple[int, ...]:
    """Return sentence-relative semantic-head indices for one token span."""

    policy = SemanticHeadRules() if rules is None else rules
    if not isinstance(policy, SemanticHeadRules):
        raise TypeError("rules must be a SemanticHeadRules instance or None.")

    n_tokens = len(token_ids)
    fields = {
        "head_token_ids": head_token_ids,
        "dependencies": dependencies,
        "pos": pos,
        "text": text,
    }
    bad_lengths = {
        name: len(values)
        for name, values in fields.items()
        if len(values) != n_tokens
    }
    if bad_lengths:
        raise ValueError(
            "Semantic-head token fields must have equal lengths; "
            f"token_ids={n_tokens}, mismatches={bad_lengths}."
        )
    if start < 0 or end > n_tokens or start >= end:
        raise ValueError(f"Invalid token span [{start}, {end}) for {n_tokens} tokens")

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

    function_pos = set(policy.function_pos)

    def nearest_content_descendant(index: int) -> int:
        queue = list(children.get(index, ()))
        visited: set[int] = set()
        while queue:
            candidate = queue.pop(0)
            if candidate in visited:
                continue
            visited.add(candidate)
            candidate_pos = None if pos[candidate] is None else str(pos[candidate]).upper()
            if dependencies[candidate] != "punct" and candidate_pos not in function_pos:
                return candidate
            queue.extend(children.get(candidate, ()))
        return index

    root_pos = None if pos[root] is None else str(pos[root]).upper()
    if policy.move_off_function_pos and root_pos in function_pos:
        root = nearest_content_descendant(root)

    if (
        policy.rewrite_quantifier_links
        and str(text[root]).casefold() in set(policy.quantifiers)
    ):
        link_words = set(policy.quantifier_links)
        link_children = [
            child
            for child in children.get(root, ())
            if str(text[child]).casefold() in link_words
        ]
        if link_children:
            replacement = nearest_content_descendant(link_children[0])
            replacement_pos = (
                None if pos[replacement] is None else str(pos[replacement]).upper()
            )
            if replacement != link_children[0] or replacement_pos not in function_pos:
                root = replacement

    heads = [root]
    excluded_roles = set(policy.coordination_excluded_roles)
    if policy.preserve_coordination and str(role).upper() not in excluded_roles:
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
