"""Feature-to-dictionary matching shared by translation and analysis."""

from __future__ import annotations

import fnmatch
import re

import numpy as np
import pandas as pd
from scipy import sparse

from text_analysis_lab.dictionaries.dictionary import Dictionary
from text_analysis_lab.dictionaries.valence import ValenceDictionary


def category_membership(
    features: list[str], dictionary: Dictionary
) -> tuple[tuple[str, ...], sparse.csr_matrix]:
    """Return ``(keys, feature x key binary membership matrix)``."""
    keys = dictionary.keys
    n_features = len(features)
    n_keys = len(keys)
    if not n_features:
        return keys, sparse.csr_matrix((0, n_keys), dtype=float)

    normalized_features = _normalized_feature_series(
        features, case_sensitive=dictionary.case_sensitive
    )
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    for key_index, key in enumerate(keys):
        mask = _match_any(
            normalized_features,
            dictionary.entries[key],
            valuetype=dictionary.valuetype,
            case_sensitive=dictionary.case_sensitive,
        )
        matched = np.flatnonzero(mask)
        if matched.size:
            rows.append(matched.astype(np.int64, copy=False))
            cols.append(np.full(matched.size, key_index, dtype=np.int64))

    if not rows:
        return keys, sparse.csr_matrix((n_features, n_keys), dtype=float)
    row_indices = np.concatenate(rows)
    col_indices = np.concatenate(cols)
    values = np.ones(row_indices.size, dtype=float)
    matrix = sparse.coo_matrix(
        (values, (row_indices, col_indices)), shape=(n_features, n_keys)
    ).tocsr()
    return keys, matrix


def valence_vectors(
    features: list[str], dictionary: ValenceDictionary
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return ``dimension -> (score vector, matched-feature mask)``."""
    normalized_features = _normalized_feature_series(
        features, case_sensitive=dictionary.case_sensitive
    )
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    if dictionary.valuetype == "fixed":
        feature_lookup: dict[str, list[int]] = {}
        for index, value in enumerate(normalized_features.tolist()):
            feature_lookup.setdefault(str(value), []).append(index)

        for dimension, pattern_values in dictionary.values.items():
            scores = np.zeros(len(features), dtype=float)
            matched = np.zeros(len(features), dtype=bool)
            for raw_pattern, value in pattern_values.items():
                pattern = (
                    str(raw_pattern)
                    if dictionary.case_sensitive
                    else str(raw_pattern).casefold()
                )
                for index in feature_lookup.get(pattern, ()):
                    if matched[index] and not np.isclose(scores[index], float(value)):
                        raise ValueError(
                            "Valence patterns assign conflicting values to the same "
                            f"matrix feature in dimension {dimension!r}: "
                            f"{features[index]!r}."
                        )
                    scores[index] = float(value)
                    matched[index] = True
            result[dimension] = (scores, matched)
        return result

    for dimension, pattern_values in dictionary.values.items():
        scores = np.zeros(len(features), dtype=float)
        matched = np.zeros(len(features), dtype=bool)
        for raw_pattern, value in pattern_values.items():
            mask = _match_any(
                normalized_features,
                (raw_pattern,),
                valuetype=dictionary.valuetype,
                case_sensitive=dictionary.case_sensitive,
            )
            if not np.any(mask):
                continue
            conflict = matched & mask & ~np.isclose(scores, float(value))
            if np.any(conflict):
                examples = [features[index] for index in np.flatnonzero(conflict)[:5]]
                raise ValueError(
                    "Valence patterns assign conflicting values to the same matrix "
                    f"feature in dimension {dimension!r}: {examples}."
                )
            scores[mask] = float(value)
            matched |= mask
        result[dimension] = (scores, matched)
    return result


def _normalized_feature_series(
    features: list[str], *, case_sensitive: bool
) -> pd.Series:
    series = pd.Series(features, dtype="string")
    return series if case_sensitive else series.str.casefold()


def _normalize_patterns(
    patterns: tuple[str, ...], *, case_sensitive: bool
) -> tuple[str, ...]:
    if case_sensitive:
        return tuple(str(pattern) for pattern in patterns)
    return tuple(str(pattern).casefold() for pattern in patterns)


def _match_any(
    features: pd.Series,
    patterns: tuple[str, ...],
    *,
    valuetype: str,
    case_sensitive: bool,
) -> np.ndarray:
    patterns = _normalize_patterns(patterns, case_sensitive=case_sensitive)
    if not patterns:
        return np.zeros(len(features), dtype=bool)

    if valuetype == "fixed":
        return features.isin(patterns).to_numpy(dtype=bool)
    if valuetype == "glob":
        translated = tuple(_glob_to_regex(pattern) for pattern in patterns)
        return _match_regex_chunks(features, translated, flags=0, contains=False)
    if valuetype == "non_whitespace_glob":
        translated = tuple(_translate_non_whitespace_glob(pattern) for pattern in patterns)
        return _match_regex_chunks(features, translated, flags=0, contains=False)
    if valuetype == "regex":
        flags = 0 if case_sensitive else re.IGNORECASE
        return _match_regex_chunks(features, patterns, flags=flags, contains=True)
    raise ValueError(f"Unsupported dictionary valuetype: {valuetype!r}.")


def _glob_to_regex(pattern: str) -> str:
    """Translate ordinary shell-style glob syntax into a whole-feature regex."""
    return fnmatch.translate(pattern)


def _translate_non_whitespace_glob(pattern: str) -> str:
    """Translate glob syntax while keeping ``*`` and ``?`` within one token."""
    pieces: list[str] = []
    add = pieces.append
    i = 0
    n = len(pattern)

    while i < n:
        char = pattern[i]
        i += 1

        if char == "*":
            # Consecutive stars are equivalent; compress them like fnmatch.
            while i < n and pattern[i] == "*":
                i += 1
            add(r"\S*")
            continue

        if char == "?":
            add(r"\S")
            continue

        if char != "[":
            add(re.escape(char))
            continue

        # Mirror fnmatch's character-class parsing for ranges, negation, and a
        # literal leading ']'. Only '*' and '?' receive the whitespace rule.
        j = i
        if j < n and pattern[j] == "!":
            j += 1
        if j < n and pattern[j] == "]":
            j += 1
        while j < n and pattern[j] != "]":
            j += 1

        if j >= n:
            add(r"\[")
            continue

        stuff = pattern[i:j]
        if "-" not in stuff:
            stuff = stuff.replace("\\", r"\\")
        else:
            chunks: list[str] = []
            k = i + 2 if i < n and pattern[i] == "!" else i + 1
            range_start = i
            while True:
                k = pattern.find("-", k, j)
                if k < 0:
                    break
                chunks.append(pattern[range_start:k])
                range_start = k + 1
                k += 3
            chunk = pattern[range_start:j]
            if chunk:
                chunks.append(chunk)
            elif chunks:
                chunks[-1] += "-"

            # Remove invalid descending ranges just as fnmatch does.
            for k in range(len(chunks) - 1, 0, -1):
                if chunks[k - 1] and chunks[k] and chunks[k - 1][-1] > chunks[k][0]:
                    chunks[k - 1] = chunks[k - 1][:-1] + chunks[k][1:]
                    del chunks[k]
            stuff = "-".join(
                value.replace("\\", r"\\").replace("-", r"\-")
                for value in chunks
            )

        # Escape regex set-operation syntax that has no glob meaning.
        stuff = re.sub(r"([&~|])", r"\\\1", stuff)
        i = j + 1
        if not stuff:
            add(r"(?!)")
        elif stuff == "!":
            add(".")
        else:
            if stuff[0] == "!":
                stuff = "^" + stuff[1:]
            elif stuff[0] in ("^", "["):
                stuff = "\\" + stuff
            add(f"[{stuff}]")

    # Glob semantics match the complete feature. The wrapper mirrors
    # fnmatch.translate() while remaining compatible with Python 3.10.
    return rf"(?s:{''.join(pieces)})\Z"


def _match_regex_chunks(
    features: pd.Series,
    patterns: tuple[str, ...],
    *,
    flags: int,
    contains: bool,
    chunk_size: int = 200,
) -> np.ndarray:
    # Dictionary regex semantics are Python-regex semantics. Pandas may back its
    # ``string`` dtype with PyArrow, whose RE2 engine rejects valid Python
    # constructs emitted by ``fnmatch.translate()`` (notably ``\Z``) and valid
    # user regex constructs such as lookarounds. Coerce once to object strings so
    # every valuetype uses the same Python ``re`` engine regardless of the
    # caller's pandas string-storage backend. Matching happens over vocabulary
    # features, not corpus rows, so this bounded conversion is inexpensive.
    python_features = features.astype(object)

    mask = np.zeros(len(features), dtype=bool)
    for start in range(0, len(patterns), chunk_size):
        chunk = patterns[start : start + chunk_size]
        combined = "(?:" + ")|(?:".join(chunk) + ")"
        if contains:
            current = python_features.str.contains(
                combined, regex=True, flags=flags, na=False
            ).to_numpy(dtype=bool)
        else:
            current = python_features.str.match(
                combined, flags=flags, na=False
            ).to_numpy(dtype=bool)
        mask |= current
    return mask
