"""Reusable, provenance-validated variance context for GD threshold indicators.

Validate once with the canonical survey-design dispatch; then evaluate the
same verified HT total variance formula on many different residual vectors.
This prevents O(number of thresholds) sampling-artifact/database reloads.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable

import numpy as np

from text_analysis_lab.core.errors import ArtifactError


@dataclass(frozen=True)
class ThresholdDesignContext:
    design: str | None
    note: str | None
    variance_total: Callable[[np.ndarray], float | None]


def prepare_threshold_design(*, surrogate, pi_artifact, audit_frame,
                             population_keys, primary_key, inclusion,
                             design=None, strata=None) -> ThresholdDesignContext:
    from text_analysis_lab.analysis.generalized_difference import (
        _design_variance_from_pi_provenance, _frame, _key_tuples, _validate_unique,
    )

    inclusion = np.asarray(inclusion, dtype=float)
    n, N = len(inclusion), len(population_keys)
    # Zero residuals only VALIDATE the design; this one-time output is not used
    # as a CDF variance. The actual threshold residuals are evaluated below.
    _, actual, note = _design_variance_from_pi_provenance(
        surrogate=surrogate, pi_artifact=pi_artifact,
        audit_frame=audit_frame, residual=np.zeros(n, dtype=float),
        population_keys=population_keys, primary_key=primary_key,
        inclusion=inclusion, design=design, strata=strata,
    )
    if note is not None and any(p in note for p in (
        "not the sampled frame", "Audit keys disagree with the recorded sample",
        "Audit size contradicts",
    )):
        raise ArtifactError(note)

    def unavailable(residual):
        return None

    if actual is None:
        return ThresholdDesignContext(None, note, unavailable)

    if actual == "srswor":
        def srs_variance(residual):
            if n == N:
                return 0.0
            return float(N * N * (1.0 - n / N) *
                         np.var(np.asarray(residual, dtype=float), ddof=1) / n)
        return ThresholdDesignContext(actual, note, srs_variance)

    if actual == "bernoulli":
        factors = (1.0 - inclusion) / (inclusion * inclusion)
        def bernoulli_variance(residual):
            r = np.asarray(residual, dtype=float)
            return float(np.sum(factors * r * r))
        return ThresholdDesignContext(actual, note, bernoulli_variance)

    assert actual == "stratified_srswor"
    if strata is None:
        operation_id = getattr(pi_artifact, "operation_id", None)
        if operation_id is None:
            raise ArtifactError("Verified stratified audit has no strata source.")
        sources = surrogate.project.operation_sources(str(operation_id))
        ids = {str(x["source_label"]): str(x["source_artifact_id"]) for x in sources}
        strata = surrogate.project.get_artifact(ids["strata"])
    fields = [str(f) for f in strata.get_data_columns()]
    if len(fields) != 1:
        raise ArtifactError("Stratified design must expose one stratum data field.")
    field = fields[0]
    strata_frame = _frame(strata, field)
    _validate_unique(strata_frame, primary_key, role="sampling strata")
    # Use the same audit ordering as residual; never rely on frame row order.
    audited = audit_frame.merge(
        strata_frame.loc[:, [*primary_key, field]],
        on=list(primary_key), validate="one_to_one", sort=False,
    )
    population_sizes = strata_frame.groupby(field, dropna=False).size().to_dict()
    groups = []
    for value, size in population_sizes.items():
        mask = (audited[field] == value).to_numpy()
        count = int(np.sum(mask))
        groups.append((mask, int(size), count))

    def strat_variance(residual):
        r = np.asarray(residual, dtype=float)
        total = 0.0
        for mask, Nh, nh in groups:
            if nh == Nh:
                continue
            total += Nh * Nh * (1.0 - nh / Nh) * float(
                np.var(r[mask], ddof=1)
            ) / nh
        return float(total)

    return ThresholdDesignContext(actual, note, strat_variance)
