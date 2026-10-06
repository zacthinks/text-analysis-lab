"""JSON-driven semantic-head resolution over dependency-aligned token spans."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any


DEFAULT_SEMANTIC_HEAD_RULES = {
    "schema_version": 1,
    "name": "bagofideas-default-semantic-head-rules",
    "description": (
        "Ordered semantic-head movement rules. The initial selection is always "
        "the syntactic head. Rule order is priority. Fields within a condition "
        "are AND; values within an array are OR. A rule moves only when its "
        "entire dependency path exists."
    ),
    "rules": [
        {
            "id": "causal_thanks_to",
            "when": {"role": ["ARGM-CAU"], "lemma": ["thank", "thanks"]},
            "move": [
                {"within": {"lemma": ["to"]}},
                {"child": {"dep": ["pobj", "obj", "obl"]}},
            ],
        },
        {
            "id": "causal_owing_to",
            "when": {"role": ["ARGM-CAU"], "lemma": ["owe", "owing"]},
            "move": [
                {"within": {"lemma": ["to"]}},
                {"child": {"dep": ["pobj", "obj", "obl"]}},
            ],
        },
        {
            "id": "causal_due_to",
            "when": {"role": ["ARGM-CAU"], "lemma": ["due"]},
            "move": [
                {"within": {"lemma": ["to"]}},
                {"child": {"dep": ["pobj", "obj", "obl"]}},
            ],
        },
        {
            "id": "causal_because_of",
            "when": {"role": ["ARGM-CAU"], "lemma": ["because"]},
            "move": [
                {"within": {"lemma": ["of"]}},
                {"child": {"dep": ["pobj", "obj", "obl"]}},
            ],
        },
        {
            "id": "causal_result_of",
            "when": {
                "role": ["ARGM-CAU"],
                "lemma": ["result", "consequence"],
            },
            "move": [
                {"within": {"lemma": ["of"]}},
                {"child": {"dep": ["pobj", "obj", "obl"]}},
            ],
        },
        {
            "id": "manner_by_means_of",
            "when": {"role": ["ARGM-MNR"], "lemma": ["mean", "means"]},
            "move": [
                {"within": {"lemma": ["of"]}},
                {"child": {"dep": ["pobj", "obj", "obl"]}},
            ],
        },
        {
            "id": "quantifier_of_object",
            "when": {
                "lemma": [
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
                ]
            },
            "move": [
                {"child": {"lemma": ["of"]}},
                {"child": {"dep": ["pobj", "obj", "obl"]}},
            ],
        },
        {
            "id": "collection_quantifier_of_object",
            "when": {
                "lemma": [
                    "pair",
                    "couple",
                    "trio",
                    "group",
                    "bunch",
                    "set",
                    "collection",
                    "series",
                    "handful",
                    "number",
                ]
            },
            "move": [
                {"child": {"lemma": ["of"]}},
                {"child": {"dep": ["pobj", "obj", "obl"]}},
            ],
        },
        {
            "id": "preposition_object",
            "when": {"pos": ["ADP", "SCONJ"]},
            "move": [{"child": {"dep": ["pobj"]}}],
        },
    ],
}

_CONDITION_FIELDS = frozenset(
    {"source", "role", "text", "lemma", "pos", "dep", "ent_type"}
)
_MOVE_KINDS = frozenset({"child", "parent", "within"})


@dataclass(frozen=True, slots=True)
class SemanticHeadMove:
    """One atomic movement in a semantic-head resolution trace."""

    rule_id: str | None
    kind: str
    from_index: int
    to_index: int


@dataclass(frozen=True, slots=True)
class SemanticHeadResolution:
    """One resolved semantic head with row-level scientific provenance."""

    syntactic_root_index: int
    semantic_head_index: int
    rule_id: str | None
    resolution_path: tuple[SemanticHeadMove, ...]
    rules_fingerprint: str


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def get_default_semantic_head_rules() -> dict[str, Any]:
    """Return an editable copy of the built-in Bag of Ideas rule document."""

    return copy.deepcopy(DEFAULT_SEMANTIC_HEAD_RULES)


def export_default_semantic_head_rules(
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Export the built-in rule library as ordinary JSON."""

    output = Path(path)
    if output.exists() and not overwrite:
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(get_default_semantic_head_rules(), indent=2) + "\n",
        encoding="utf-8",
    )
    return output


@dataclass(frozen=True, slots=True)
class SemanticHeadRules:
    """Validated, serializable Bag of Ideas semantic-head rule library."""

    schema_version: int = 1
    name: str = "bagofideas-default-semantic-head-rules"
    description: str = DEFAULT_SEMANTIC_HEAD_RULES["description"]
    rules: tuple[Mapping[str, Any], ...] = field(
        default_factory=lambda: tuple(
            copy.deepcopy(DEFAULT_SEMANTIC_HEAD_RULES["rules"])
        )
    )

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError(
                "Semantic-head rule schema_version must currently equal 1."
            )
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Semantic-head rule name must be a non-empty string.")
        if not isinstance(self.description, str):
            raise TypeError("Semantic-head rule description must be a string.")

        normalized: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for index, raw_rule in enumerate(self.rules):
            rule = _validate_rule(raw_rule, index=index)
            rule_id = rule["id"]
            if rule_id in seen_ids:
                raise ValueError(f"Duplicate semantic-head rule id {rule_id!r}.")
            seen_ids.add(rule_id)
            normalized.append(rule)
        object.__setattr__(self, "rules", tuple(_freeze_json(rule) for rule in normalized))

    @property
    def fingerprint(self) -> str:
        """Stable SHA-256 fingerprint of the complete JSON rule document."""

        payload = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Return the original BoI JSON-compatible rule document shape."""

        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "description": self.description,
            "rules": [_thaw_json(rule) for rule in self.rules],
        }

    @classmethod
    def from_dict(cls, state: Mapping[str, Any]) -> "SemanticHeadRules":
        """Restore a validated rule library from the BoI JSON document shape."""

        if not isinstance(state, Mapping):
            raise TypeError("Semantic-head rules must be a mapping.")
        allowed = {"schema_version", "name", "description", "rules"}
        unknown = set(state) - allowed
        if unknown:
            raise ValueError(
                f"Unknown semantic-head rule document fields: {sorted(unknown)}."
            )
        values = dict(state)
        if "rules" in values:
            raw_rules = values["rules"]
            if not isinstance(raw_rules, Sequence) or isinstance(
                raw_rules, (str, bytes)
            ):
                raise TypeError("Semantic-head rules must be a JSON array.")
            values["rules"] = tuple(raw_rules)
        return cls(**values)

    @classmethod
    def from_json_file(cls, path: str | Path) -> "SemanticHeadRules":
        """Load an editable BoI semantic-head rule JSON file."""

        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise ValueError("Semantic-head rule JSON must contain an object.")
        return cls.from_dict(value)


def load_semantic_head_rules(
    value: SemanticHeadRules | Mapping[str, Any] | str | Path | None = None,
) -> SemanticHeadRules:
    """Normalize the same rule inputs accepted by the original BoI package."""

    if value is None:
        return SemanticHeadRules.from_dict(get_default_semantic_head_rules())
    if isinstance(value, SemanticHeadRules):
        return value
    if isinstance(value, Mapping):
        return SemanticHeadRules.from_dict(value)
    if isinstance(value, (str, Path)):
        return SemanticHeadRules.from_json_file(value)
    raise TypeError(
        "Semantic-head rules must be SemanticHeadRules, a mapping, a JSON path, "
        "or None."
    )


def resolve_semantic_heads(
    *,
    start: int,
    end: int,
    token_ids: Sequence[int],
    head_token_ids: Sequence[int | None],
    dependencies: Sequence[str | None],
    pos: Sequence[str | None],
    text: Sequence[str],
    role: str | None,
    lemmas: Sequence[str | None] | None = None,
    ent_types: Sequence[str | None] | None = None,
    source: str | None = None,
    rules: SemanticHeadRules | Mapping[str, Any] | str | Path | None = None,
) -> tuple[SemanticHeadResolution, ...]:
    """Resolve semantic heads with movement provenance for one token span.

    This implements the original Bag of Ideas rule semantics: start at the
    syntactic head, evaluate ordered rules, require a complete move path, and
    restart from rule 1 after each successful move. Coordination expansion is
    engine behavior rather than user rule syntax.
    """

    policy = load_semantic_head_rules(rules)
    n_tokens = len(token_ids)
    lemma_values = (
        [str(value) for value in text]
        if lemmas is None
        else list(lemmas)
    )
    ent_values = (
        [None] * n_tokens if ent_types is None else list(ent_types)
    )
    fields = {
        "head_token_ids": head_token_ids,
        "dependencies": dependencies,
        "pos": pos,
        "text": text,
        "lemmas": lemma_values,
        "ent_types": ent_values,
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

    global_to_local = {
        int(token_id): index for index, token_id in enumerate(token_ids)
    }
    span_indices = tuple(range(start, end))
    span_set = set(span_indices)
    children: dict[int, list[int]] = {index: [] for index in span_indices}
    for child in span_indices:
        head = head_token_ids[child]
        if head is None:
            continue
        parent = global_to_local.get(int(head))
        if parent in span_set and parent != child:
            children[parent].append(child)

    roots = [
        index
        for index in span_indices
        if head_token_ids[index] is None
        or global_to_local.get(int(head_token_ids[index]), -1) not in span_set
        or global_to_local.get(int(head_token_ids[index]), -1) == index
    ]
    non_punct_roots = [
        index for index in roots if _fold(dependencies[index]) != "punct"
    ]
    syntactic_root = (non_punct_roots or roots or [start])[0]

    context = {
        "source": source,
        "role": role,
    }

    def attributes(index: int) -> dict[str, str | None]:
        return {
            "source": source,
            "role": role,
            "text": str(text[index]),
            "lemma": None
            if lemma_values[index] is None
            else str(lemma_values[index]),
            "pos": None if pos[index] is None else str(pos[index]),
            "dep": None
            if dependencies[index] is None
            else str(dependencies[index]),
            "ent_type": None
            if ent_values[index] is None
            else str(ent_values[index]),
        }

    def matches(index: int, condition: Mapping[str, Any]) -> bool:
        return _matches_condition(
            condition,
            values={**context, **attributes(index)},
        )

    def follow_step(
        current: int,
        step: Mapping[str, Any],
    ) -> int | None:
        kind, selector = next(iter(step.items()))
        if kind == "child":
            for candidate in children.get(current, ()):
                if matches(candidate, selector):
                    return candidate
            return None
        if kind == "parent":
            head = head_token_ids[current]
            if head is None:
                return None
            candidate = global_to_local.get(int(head))
            if candidate not in span_set or candidate == current:
                return None
            return candidate if matches(candidate, selector) else None
        if kind == "within":
            for candidate in span_indices:
                if candidate != current and matches(candidate, selector):
                    return candidate
            return None
        raise AssertionError(f"Unexpected semantic-head move kind {kind!r}.")

    def coordinated_children(index: int) -> tuple[int, ...]:
        # Predicate spans intentionally retain a single predicate head. For
        # argument/adjunct/coreference spans, coordination creates independent
        # semantic-head branches.
        if _fold(role) == "v":
            return ()
        return tuple(
            candidate
            for candidate in span_indices
            if _fold(dependencies[candidate]) == "conj"
            and head_token_ids[candidate] is not None
            and global_to_local.get(int(head_token_ids[candidate])) == index
        )

    pending: list[tuple[int, tuple[SemanticHeadMove, ...]]] = [
        (syntactic_root, ())
    ]
    queued = {syntactic_root}
    resolved: list[SemanticHeadResolution] = []

    while pending:
        current, path = pending.pop(0)
        visited = {current}

        while True:
            # Coordination must branch from every head reached by the rule
            # engine, not only from the final resolved head. Record the branch
            # explicitly so a later artifact can explain why a second head
            # exists even when both branches later move under ordinary rules.
            for candidate in coordinated_children(current):
                if candidate not in queued:
                    branch_move = SemanticHeadMove(
                        rule_id=None,
                        kind="coordination",
                        from_index=current,
                        to_index=candidate,
                    )
                    pending.append((candidate, (*path, branch_move)))
                    queued.add(candidate)

            moved = False
            for rule in policy.rules:
                if not matches(current, rule["when"]):
                    continue
                candidate = current
                complete = True
                rule_moves: list[SemanticHeadMove] = []
                for step in rule["move"]:
                    kind = str(next(iter(step)))
                    next_candidate = follow_step(candidate, step)
                    if next_candidate is None:
                        complete = False
                        break
                    rule_moves.append(
                        SemanticHeadMove(
                            rule_id=str(rule["id"]),
                            kind=kind,
                            from_index=candidate,
                            to_index=next_candidate,
                        )
                    )
                    candidate = next_candidate
                if not complete or candidate == current:
                    continue
                if candidate in visited:
                    raise ValueError(
                        "Semantic-head rules produced a movement cycle at "
                        f"rule {rule['id']!r}."
                    )
                current = candidate
                path = (*path, *rule_moves)
                visited.add(current)
                moved = True
                break

            if not moved:
                last_rule = next(
                    (
                        move.rule_id
                        for move in reversed(path)
                        if move.rule_id is not None
                    ),
                    None,
                )
                resolved.append(
                    SemanticHeadResolution(
                        syntactic_root_index=syntactic_root,
                        semantic_head_index=current,
                        rule_id=last_rule,
                        resolution_path=path,
                        rules_fingerprint=policy.fingerprint,
                    )
                )
                break

    by_head: dict[int, SemanticHeadResolution] = {}
    for result in resolved:
        by_head.setdefault(result.semantic_head_index, result)
    return tuple(by_head.values())


def resolve_semantic_head_indices(
    *,
    start: int,
    end: int,
    token_ids: Sequence[int],
    head_token_ids: Sequence[int | None],
    dependencies: Sequence[str | None],
    pos: Sequence[str | None],
    text: Sequence[str],
    role: str | None,
    lemmas: Sequence[str | None] | None = None,
    ent_types: Sequence[str | None] | None = None,
    source: str | None = None,
    rules: SemanticHeadRules | Mapping[str, Any] | str | Path | None = None,
) -> tuple[int, ...]:
    """Project rich semantic-head resolutions to sentence-relative indices."""

    return tuple(
        result.semantic_head_index
        for result in resolve_semantic_heads(
            start=start,
            end=end,
            token_ids=token_ids,
            head_token_ids=head_token_ids,
            dependencies=dependencies,
            pos=pos,
            text=text,
            role=role,
            lemmas=lemmas,
            ent_types=ent_types,
            source=source,
            rules=rules,
        )
    )


def _validate_rule(raw_rule: Mapping[str, Any], *, index: int) -> dict[str, Any]:
    if not isinstance(raw_rule, Mapping):
        raise TypeError(f"Semantic-head rule {index} must be a mapping.")
    unknown = set(raw_rule) - {"id", "when", "move"}
    if unknown:
        raise ValueError(
            f"Semantic-head rule {index} has unknown fields {sorted(unknown)}."
        )
    rule_id = raw_rule.get("id")
    if not isinstance(rule_id, str) or not rule_id:
        raise ValueError(f"Semantic-head rule {index} requires a non-empty id.")
    when = raw_rule.get("when", {})
    if not isinstance(when, Mapping):
        raise TypeError(f"Semantic-head rule {rule_id!r} when must be an object.")
    normalized_when = _validate_condition(when, where=f"rule {rule_id!r} when")

    move = raw_rule.get("move")
    if not isinstance(move, Sequence) or isinstance(move, (str, bytes)) or not move:
        raise ValueError(
            f"Semantic-head rule {rule_id!r} move must be a non-empty array."
        )
    normalized_move: list[dict[str, Any]] = []
    for step_index, raw_step in enumerate(move):
        if not isinstance(raw_step, Mapping) or len(raw_step) != 1:
            raise ValueError(
                f"Semantic-head rule {rule_id!r} move step {step_index} must "
                "contain exactly one of child, parent, or within."
            )
        kind, selector = next(iter(raw_step.items()))
        if kind not in _MOVE_KINDS:
            raise ValueError(
                f"Semantic-head rule {rule_id!r} uses unsupported move {kind!r}."
            )
        if not isinstance(selector, Mapping):
            raise TypeError(
                f"Semantic-head rule {rule_id!r} move selector must be an object."
            )
        normalized_move.append(
            {
                str(kind): _validate_condition(
                    selector,
                    where=f"rule {rule_id!r} move step {step_index}",
                    allow_context=False,
                )
            }
        )
    return {"id": rule_id, "when": normalized_when, "move": normalized_move}


def _validate_condition(
    condition: Mapping[str, Any],
    *,
    where: str,
    allow_context: bool = True,
) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    allowed_fields = _CONDITION_FIELDS if allow_context else (
        _CONDITION_FIELDS - {"source", "role"}
    )
    for raw_key, raw_value in condition.items():
        key = str(raw_key)
        if key in {"any", "not"}:
            if key == "any":
                if (
                    not isinstance(raw_value, Sequence)
                    or isinstance(raw_value, (str, bytes))
                    or not raw_value
                ):
                    raise ValueError(f"{where} any must be a non-empty array.")
                normalized[key] = [
                    _validate_condition(
                        value,
                        where=f"{where} any",
                        allow_context=allow_context,
                    )
                    for value in raw_value
                    if isinstance(value, Mapping)
                ]
                if len(normalized[key]) != len(raw_value):
                    raise TypeError(f"{where} any entries must be objects.")
            else:
                if not isinstance(raw_value, Mapping):
                    raise TypeError(f"{where} not must be an object.")
                normalized[key] = _validate_condition(
                    raw_value,
                    where=f"{where} not",
                    allow_context=allow_context,
                )
            continue
        if key not in allowed_fields:
            raise ValueError(f"{where} uses unsupported condition field {key!r}.")
        values = (
            list(raw_value)
            if isinstance(raw_value, Sequence)
            and not isinstance(raw_value, (str, bytes))
            else [raw_value]
        )
        if not values:
            raise ValueError(f"{where} field {key!r} cannot have an empty array.")
        normalized[key] = [None if value is None else str(value) for value in values]
    return normalized


def _matches_condition(
    condition: Mapping[str, Any],
    *,
    values: Mapping[str, str | None],
) -> bool:
    for key, expected in condition.items():
        if key == "any":
            if not any(_matches_condition(item, values=values) for item in expected):
                return False
            continue
        if key == "not":
            if _matches_condition(expected, values=values):
                return False
            continue
        actual = values.get(key)
        folded_actual = _fold(actual)
        if folded_actual not in {_fold(value) for value in expected}:
            return False
    return True


def _fold(value: Any) -> str | None:
    return None if value is None else str(value).casefold()
