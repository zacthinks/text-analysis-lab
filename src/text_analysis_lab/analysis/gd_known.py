"""Validate exactly known finite-population outcome labels for design estimation.

Known labels need no inverse-probability weights. They must be disjoint from
the probability audit, and their keys are excluded from its eligible frame.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from text_analysis_lab.core.artifact_base import BaseArtifact
from text_analysis_lab.core.errors import ArtifactError
from text_analysis_lab.analysis.generalized_difference import (
    _frame, _key_tuples, _numeric, _validate_unique,
)


def known_table(project, known, *, primary_key, label_field):
    """Read a single keyed artifact or ordered collection of disjoint artifacts.

    If a known artifact exposes one data field, infer it; otherwise use the
    named audit-label field (which must occur in that artifact). Outcome field
    names need not match when each artifact is otherwise unambiguous.
    """
    keys = tuple(primary_key)
    if known is None:
        return pd.DataFrame(columns=[*keys, "__known_y__"]), set()
    if isinstance(known, (BaseArtifact, str)):
        items = [known]
    elif isinstance(known, Sequence) and not isinstance(known, (bytes, bytearray)):
        items = list(known)
    else:
        raise TypeError("known= must be an artifact, artifact reference, or sequence.")
    frames = []
    all_keys = set()
    for index, ref in enumerate(items):
        artifact = project.get_artifact(ref)
        if tuple(str(k) for k in artifact.primary_key) != keys:
            raise ArtifactError("known= artifacts must share the audit primary keys.")
        fields = [str(c) for c in artifact.get_data_columns()]
        if label_field in fields:
            field = label_field
        elif len(fields) == 1:
            field = fields[0]
        else:
            raise ArtifactError(
                f"known= artifact {index} requires an unambiguous label field; "
                f"available data fields: {fields}."
            )
        table = _frame(artifact, field)
        _validate_unique(table, keys, role=f"known[{index}]")
        values = _numeric(table[field], name=f"known[{index}] field {field!r}")
        unit_keys = set(_key_tuples(table, keys))
        if all_keys.intersection(unit_keys):
            raise ArtifactError("known= label artifacts have overlapping keys.")
        all_keys.update(unit_keys)
        frames.append(table.loc[:, list(keys)].assign(__known_y__=values))
    return (
        pd.concat(frames, ignore_index=True)
        if frames else pd.DataFrame(columns=[*keys, "__known_y__"])
    ), all_keys


def split_target_frame(frame_keys, known_keys, recorded_keys=None):
    """Return eligible audit keys U and target keys F=U union K.

    An original TeAL frame (when present) is authoritative: a known key
    must never have been eligible for the recorded random audit. Otherwise,
    an explicitly supplied frame may be U or F; in either case observed known
    keys are excluded from the audit eligibility set. External selection
    still requires a truthful design= assertion.
    """
    observed = set(frame_keys)
    known = set(known_keys)
    if recorded_keys is not None:
        audit_eligible = set(recorded_keys)
        if known & audit_eligible:
            raise ArtifactError(
                "known= keys overlap the recorded probability-sampling frame; "
                "the original audit design included these units."
            )
        if observed - known != audit_eligible:
            raise ArtifactError(
                "The population/surrogate frame disagrees with the recorded "
                "probability-audit remainder after excluding known=."
            )
        if not known.issubset(observed | known):
            raise ArtifactError("Invalid known= frame.")
        return audit_eligible, observed | known
    return observed - known, observed | known
