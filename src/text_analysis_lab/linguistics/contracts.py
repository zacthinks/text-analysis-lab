"""Structural contracts shared by TeAL linguistic artifacts.

These checks deliberately validate computability and stable row identity, not
shared lineage.  Provenance remains inspectable, but different derivations are
not rejected merely because their lineage differs.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from text_analysis_lab.core.errors import ArtifactError


def validate_linguistic_contracts(
    *,
    sentences: pd.DataFrame,
    tokens: pd.DataFrame,
    predicates: pd.DataFrame | None = None,
    role_spans: pd.DataFrame | None = None,
    role_heads: pd.DataFrame | None = None,
    candidates: pd.DataFrame | None = None,
    senses: pd.DataFrame | None = None,
    wsd_unresolved: pd.DataFrame | None = None,
    mentions: pd.DataFrame | None = None,
    sentence_keys: Sequence[str] = ("row_id", "sentence_id"),
    token_key: str = "token_id",
) -> None:
    """Validate the structural identity contract consumed by Bag of Ideas.

    The function accepts ordinary combined key+data DataFrames.  It intentionally
    does not inspect TeAL lineage/basis metadata: structurally compatible artifacts
    remain usable even when they were produced by different derivations.
    """

    sentence_keys = tuple(str(value) for value in sentence_keys)
    if not sentence_keys:
        raise ValueError("sentence_keys must contain at least one column.")
    token_key = str(token_key)
    token_keys = (*sentence_keys, token_key)
    document_keys = sentence_keys[:-1]

    _require_columns(sentences, sentence_keys, label="sentences")
    _require_columns(
        tokens,
        (*token_keys, "text", "lemma", "pos"),
        label="tokens",
    )
    _require_unique(sentences, sentence_keys, label="sentences")
    _require_unique(tokens, token_keys, label="tokens")

    sentence_identity = _identity_set(sentences, sentence_keys)
    token_identity = _identity_set(tokens, token_keys)
    for identity in _identity_set(tokens, sentence_keys):
        if identity not in sentence_identity:
            raise ArtifactError(
                f"tokens references missing sentence identity {identity}."
            )

    if predicates is not None:
        predicate_keys = (*sentence_keys, "predicate_id")
        _require_columns(
            predicates,
            (*predicate_keys, "predicate_token_id"),
            label="predicates",
        )
        _require_unique(predicates, predicate_keys, label="predicates")
        for row in predicates.itertuples(index=False):
            sentence_identity_row = tuple(getattr(row, key) for key in sentence_keys)
            if sentence_identity_row not in sentence_identity:
                raise ArtifactError(
                    "predicates references missing sentence identity "
                    f"{sentence_identity_row}."
                )
            token_identity_row = (
                *sentence_identity_row,
                getattr(row, "predicate_token_id"),
            )
            if token_identity_row not in token_identity:
                raise ArtifactError(
                    "predicate_token_id references missing token identity "
                    f"{token_identity_row}."
                )

    if role_spans is not None:
        predicate_keys = (*sentence_keys, "predicate_id")
        role_keys = (*predicate_keys, "role_id")
        _require_columns(
            role_spans,
            (*role_keys, "token_start_id", "token_end_id"),
            label="role_spans",
        )
        _require_unique(role_spans, role_keys, label="role_spans")
        if predicates is not None:
            predicate_identity = _identity_set(predicates, predicate_keys)
            for identity in _identity_set(role_spans, predicate_keys):
                if identity not in predicate_identity:
                    raise ArtifactError(
                        f"role_spans references missing predicate identity {identity}."
                    )

        token_ids_by_sentence = _token_ids_by_sentence(
            tokens,
            sentence_keys=sentence_keys,
            token_key=token_key,
        )
        for row in role_spans.itertuples(index=False):
            sentence_identity_row = tuple(getattr(row, key) for key in sentence_keys)
            start = int(getattr(row, "token_start_id"))
            end = int(getattr(row, "token_end_id"))
            if end <= start:
                raise ArtifactError(
                    f"role_spans has invalid half-open token span [{start}, {end})."
                )
            available = token_ids_by_sentence.get(sentence_identity_row, set())
            missing = [token_id for token_id in range(start, end) if token_id not in available]
            if missing:
                raise ArtifactError(
                    "role_spans references missing token identities in sentence "
                    f"{sentence_identity_row}: {missing}."
                )

    if role_heads is not None:
        role_keys = (*sentence_keys, "predicate_id", "role_id")
        head_keys = (*role_keys, "head_id")
        _require_columns(
            role_heads,
            (
                *head_keys,
                "syntactic_root_token_id",
                "head_token_id",
                "head_text",
                "head_lemma",
                "head_pos",
            ),
            label="role_heads",
        )
        _require_unique(role_heads, head_keys, label="role_heads")
        if role_spans is not None:
            role_identity = _identity_set(role_spans, role_keys)
            for identity in _identity_set(role_heads, role_keys):
                if identity not in role_identity:
                    raise ArtifactError(
                        f"role_heads references missing role-span identity {identity}."
                    )
        for row in role_heads.itertuples(index=False):
            sentence_identity_row = tuple(getattr(row, key) for key in sentence_keys)
            for field in ("syntactic_root_token_id", "head_token_id"):
                identity = (*sentence_identity_row, getattr(row, field))
                if identity not in token_identity:
                    raise ArtifactError(
                        f"role_heads {field} references missing token identity {identity}."
                    )
            _check_token_projection(
                tokens,
                identity=(*sentence_identity_row, getattr(row, "head_token_id")),
                token_keys=token_keys,
                expected={
                    "text": getattr(row, "head_text"),
                    "lemma": getattr(row, "head_lemma"),
                    "pos": getattr(row, "head_pos"),
                },
                label="role_heads",
            )

    if candidates is not None:
        candidate_keys = (*token_keys, "candidate_id")
        _require_columns(
            candidates,
            (
                *candidate_keys,
                "surface_form",
                "parser_lemma",
                "pos",
                "sense_id",
                "rank",
                "selected",
            ),
            label="candidates",
        )
        _require_unique(candidates, candidate_keys, label="candidates")
        for row in candidates.itertuples(index=False):
            identity = tuple(getattr(row, key) for key in token_keys)
            if identity not in token_identity:
                raise ArtifactError(
                    f"candidates references missing token identity {identity}."
                )
            _check_token_projection(
                tokens,
                identity=identity,
                token_keys=token_keys,
                expected={
                    "text": getattr(row, "surface_form"),
                    "lemma": getattr(row, "parser_lemma"),
                    "pos": getattr(row, "pos"),
                },
                label="candidates",
            )

    if senses is not None:
        _require_columns(
            senses,
            (*token_keys, "surface_form", "parser_lemma", "pos", "sense_id"),
            label="senses",
        )
        _require_unique(senses, token_keys, label="senses")
        candidate_senses = (
            {
                (*tuple(getattr(row, key) for key in token_keys), str(row.sense_id))
                for row in candidates.itertuples(index=False)
            }
            if candidates is not None
            else None
        )
        for row in senses.itertuples(index=False):
            identity = tuple(getattr(row, key) for key in token_keys)
            if identity not in token_identity:
                raise ArtifactError(f"senses references missing token identity {identity}.")
            if candidate_senses is not None and (*identity, str(row.sense_id)) not in candidate_senses:
                raise ArtifactError(
                    "senses contains a sense not present among stored candidates for "
                    f"token {identity}: {row.sense_id!r}."
                )
            _check_token_projection(
                tokens,
                identity=identity,
                token_keys=token_keys,
                expected={
                    "text": getattr(row, "surface_form"),
                    "lemma": getattr(row, "parser_lemma"),
                    "pos": getattr(row, "pos"),
                },
                label="senses",
            )

    if wsd_unresolved is not None:
        _require_columns(
            wsd_unresolved,
            (*token_keys, "surface_form", "parser_lemma", "pos", "reason"),
            label="wsd_unresolved",
        )
        _require_unique(wsd_unresolved, token_keys, label="wsd_unresolved")
        unresolved_identity = _identity_set(wsd_unresolved, token_keys)
        for identity in unresolved_identity:
            if identity not in token_identity:
                raise ArtifactError(
                    f"wsd_unresolved references missing token identity {identity}."
                )
        if senses is not None:
            overlap = unresolved_identity & _identity_set(senses, token_keys)
            if overlap:
                raise ArtifactError(
                    "A token cannot be both WSD-resolved and WSD-unresolved in the "
                    f"same contract view: {sorted(overlap)!r}."
                )

    if mentions is not None:
        mention_keys = (*document_keys, "cluster_id", "mention_id")
        _require_columns(
            mentions,
            (
                *mention_keys,
                sentence_keys[-1],
                "token_start_id",
                "token_end_id",
                "head_token_id",
                "head_text",
                "head_lemma",
                "head_pos",
            ),
            label="mentions",
        )
        _require_unique(mentions, mention_keys, label="mentions")
        token_ids_by_sentence = _token_ids_by_sentence(
            tokens,
            sentence_keys=sentence_keys,
            token_key=token_key,
        )
        for row in mentions.itertuples(index=False):
            sentence_identity_row = tuple(
                getattr(row, key)
                for key in (*document_keys, sentence_keys[-1])
            )
            start = int(getattr(row, "token_start_id"))
            end = int(getattr(row, "token_end_id"))
            if end <= start:
                raise ArtifactError(
                    f"mentions has invalid half-open token span [{start}, {end})."
                )
            available = token_ids_by_sentence.get(sentence_identity_row, set())
            missing = [token_id for token_id in range(start, end) if token_id not in available]
            if missing:
                raise ArtifactError(
                    "mentions references missing token identities in sentence "
                    f"{sentence_identity_row}: {missing}."
                )
            head_identity = (*sentence_identity_row, getattr(row, "head_token_id"))
            if head_identity not in token_identity:
                raise ArtifactError(
                    f"mentions head_token_id references missing token identity {head_identity}."
                )
            _check_token_projection(
                tokens,
                identity=head_identity,
                token_keys=token_keys,
                expected={
                    "text": getattr(row, "head_text"),
                    "lemma": getattr(row, "head_lemma"),
                    "pos": getattr(row, "head_pos"),
                },
                label="mentions",
            )


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], *, label: str) -> None:
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise ArtifactError(f"{label} is missing required columns {missing}.")


def _require_unique(frame: pd.DataFrame, keys: Sequence[str], *, label: str) -> None:
    if frame.duplicated(list(keys)).any():
        raise ArtifactError(f"{label} contains duplicate primary-key identities {list(keys)}.")


def _identity_set(frame: pd.DataFrame, keys: Sequence[str]) -> set[tuple[object, ...]]:
    return {
        tuple(row)
        for row in frame.loc[:, list(keys)].itertuples(index=False, name=None)
    }


def _token_ids_by_sentence(
    tokens: pd.DataFrame,
    *,
    sentence_keys: Sequence[str],
    token_key: str,
) -> dict[tuple[object, ...], set[int]]:
    result: dict[tuple[object, ...], set[int]] = {}
    for raw_key, group in tokens.groupby(list(sentence_keys), sort=False, dropna=False):
        key = raw_key if isinstance(raw_key, tuple) else (raw_key,)
        result[tuple(key)] = set(group[token_key].astype(int).tolist())
    return result


def _check_token_projection(
    tokens: pd.DataFrame,
    *,
    identity: tuple[object, ...],
    token_keys: Sequence[str],
    expected: dict[str, object],
    label: str,
) -> None:
    mask = pd.Series(True, index=tokens.index)
    for key, value in zip(token_keys, identity, strict=True):
        mask &= tokens[key] == value
    row = tokens.loc[mask]
    if len(row) != 1:
        raise ArtifactError(
            f"{label} token projection does not resolve uniquely for identity {identity}."
        )
    source = row.iloc[0]
    for source_field, expected_value in expected.items():
        source_value = source[source_field]
        if pd.isna(source_value) and pd.isna(expected_value):
            continue
        if str(source_value) != str(expected_value):
            raise ArtifactError(
                f"{label} disagrees with token {identity} on {source_field!r}: "
                f"{expected_value!r} != {source_value!r}."
            )
