"""Finite-population generalized-difference CDF and median estimates.

Pointwise threshold estimates use Horvitz-Thompson GD correction. The
resulting curve is projected onto valid CDF ordinates before left inversion;
projection changes the estimator and does not preserve thresholdwise
design-unbiasedness. Median uncertainty is deliberately not borrowed from
mean or threshold-CDF standard errors.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import NormalDist

import numpy as np
import pandas as pd

from text_analysis_lab.core.errors import ArtifactError


def bounded_isotonic_cdf(raw: np.ndarray) -> np.ndarray:
    """Equal-knot-weight L2 isotonic projection, with fixed final CDF=1.

    The support is a sorted distinct numerical grid. There is a CDF value
    zero strictly below its first knot; the final knot is constrained to one.
    """
    x = np.asarray(raw, dtype=float)
    if x.ndim != 1 or not len(x) or not np.all(np.isfinite(x)):
        raise ArtifactError("Raw threshold CDF must be nonempty and finite.")
    if len(x) == 1:
        return np.asarray([1.0])
    # Pool-adjacent-violators; one unit of weight per unique threshold.
    totals: list[float] = []
    counts: list[int] = []
    for value in x[:-1]:
        totals.append(float(value))
        counts.append(1)
        while len(totals) > 1 and totals[-2] / counts[-2] > totals[-1] / counts[-1]:
            totals[-2] += totals[-1]
            counts[-2] += counts[-1]
            totals.pop()
            counts.pop()
    fitted = np.asarray([
        min(1.0, max(0.0, total / count))
        for total, count in zip(totals, counts)
        for _ in range(count)
    ], dtype=float)
    return np.concatenate([fitted, [1.0]])


@dataclass(frozen=True)
class GeneralizedDifferenceQuantileResult:
    """Corrected-CDF median with transparent pointwise diagnostics."""

    statistic: str
    estimate: float
    quantile_level: float
    population_size: int
    audit_size: int
    inference: str
    variance_design: str | None
    variance_note: str
    projection_method: str
    _cdf: pd.DataFrame
    standard_error: float | None = None

    @property
    def estimated_median(self) -> float:
        return self.estimate

    def cdf_frame(self, *, level: float | None = None) -> pd.DataFrame:
        """Return raw/adjusted pointwise CDF and SE, optionally normal limits.

        The normal bounds are pointwise approximation diagnostics; they are
        NOT a simultaneous confidence band or a quantile confidence interval.
        They are intentionally not clipped to [0,1].
        """
        frame = self._cdf.copy()
        if level is not None:
            if not (0.0 < level < 1.0):
                raise ValueError("level must satisfy 0 < level < 1.")
            z = NormalDist().inv_cdf(0.5 + level / 2)
            frame["pointwise_lower"] = frame["raw_cdf"] - z * frame["cdf_se"]
            frame["pointwise_upper"] = frame["raw_cdf"] + z * frame["cdf_se"]
        return frame

    def confidence_interval(self, *, level: float = 0.95) -> tuple[float, float]:
        if not (0.0 < level < 1.0):
            raise ValueError("level must satisfy 0 < level < 1.")
        if self.standard_error == 0.0:
            return self.estimate, self.estimate
        raise ValueError(
            "GD quantile confidence interval unavailable for a partial audit: "
            "only exact finite-census intervals are supported; raw CDF "
            "and pointwise design uncertainty remain available."
        )

    def p_value(self, *, null: float = 0.0) -> float:
        raise ValueError("GD quantile hypothesis tests require separate inference.")

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "statistic": self.statistic, "estimate": self.estimate,
            "quantile_level": self.quantile_level,
            "population_size": self.population_size,
            "audit_size": self.audit_size,
            "inference": self.inference,
            "variance_design": self.variance_design,
            "variance_note": self.variance_note,
            "projection_method": self.projection_method,
            "standard_error": self.standard_error,
        }])


def generalized_difference_median(
    surrogate, *, gold, pi,
    surrogate_field: str, gold_field: str, pi_field: str,
    design: str | None, strata, quantile: float = 0.5,
    statistic: str = "median",
) -> GeneralizedDifferenceQuantileResult:
    """Compute thresholdwise GD CDF followed by bounded isotonic inversion."""
    from text_analysis_lab.analysis.generalized_difference import (
        _frame, _key_tuples,
        _numeric, _require_field, _validate_unique,
    )
    from text_analysis_lab.analysis.gd_design_variance import check_design
    check_design(design)
    project = surrogate.project
    gold_artifact = project.get_artifact(gold)
    pi_artifact = project.get_artifact(pi)
    strata_artifact = None if strata is None else project.get_artifact(strata)

    keys = tuple(str(k) for k in surrogate.primary_key)
    if not keys:
        raise ArtifactError("Surrogate has no primary-key columns.")
    for role, artifact in (("gold", gold_artifact), ("pi", pi_artifact)):
        if tuple(str(k) for k in artifact.primary_key) != keys:
            raise ArtifactError(f"{role} and surrogate primary-key columns differ.")
    for artifact, field, role in (
        (surrogate, surrogate_field, "surrogate"),
        (gold_artifact, gold_field, "gold"),
        (pi_artifact, pi_field, "pi"),
    ):
        _require_field(artifact, field, role=role)

    pop = _frame(surrogate, surrogate_field)
    audit = _frame(gold_artifact, gold_field)
    probabilities = _frame(pi_artifact, pi_field)
    for name, frame in (("surrogate", pop), ("gold", audit), ("pi", probabilities)):
        _validate_unique(frame, keys, role=name)
    if len(pop) == 0 or len(audit) == 0:
        raise ArtifactError("GD median requires a nonempty frame and probability audit.")
    full_keys = set(_key_tuples(pop, keys))
    audit_keys = set(_key_tuples(audit, keys))
    if set(_key_tuples(probabilities, keys)) != audit_keys:
        raise ArtifactError("pi and gold must have identical audit key sets.")
    if not audit_keys.issubset(full_keys):
        raise ArtifactError("GD median audit contains keys outside the surrogate frame.")

    # Namespace both joined values so even identical user field names do not
    # produce pandas _x/_y suffix collisions.
    reserved = set(keys) | {gold_field, surrogate_field, pi_field}
    proxy_col = "__teal_gd_quantile_proxy__"
    weight_col = "__teal_gd_quantile_pi__"
    while proxy_col in reserved:
        proxy_col += "_"
    reserved.add(proxy_col)
    while weight_col in reserved:
        weight_col += "_"
    joined = (
        audit.merge(
            pop.loc[:, [*keys, surrogate_field]].rename(
                columns={surrogate_field: proxy_col}
            ),
            on=list(keys), validate="one_to_one", sort=False,
        ).merge(
            probabilities.rename(columns={pi_field: weight_col}),
            on=list(keys), validate="one_to_one", sort=False,
        )
    )
    y = _numeric(joined[gold_field], name="gold outcomes")
    q_audit = _numeric(joined[proxy_col], name="audited surrogate outcomes")
    q_all = _numeric(pop[surrogate_field], name="full-frame surrogate outcomes")
    inclusion = _numeric(joined[weight_col], name="pi")
    if np.any(inclusion <= 0.0) or np.any(inclusion > 1.0):
        raise ArtifactError("pi must satisfy 0 < pi <= 1.")

    knots = np.unique(np.concatenate([q_all, y]))
    N = len(q_all)
    inverse = 1.0 / inclusion
    if not np.all(np.isfinite(inverse)):
        raise ArtifactError("Inverse inclusion probabilities must be finite.")

    from text_analysis_lab.analysis.gd_quantile_design import prepare_threshold_design
    validated = prepare_threshold_design(
        surrogate=surrogate, pi_artifact=pi_artifact,
        audit_frame=joined.loc[:, list(keys)].copy(),
        population_keys=full_keys, primary_key=keys,
        inclusion=inclusion, design=design, strata=strata_artifact,
    )

    raw: list[float] = []
    se: list[float] = []
    notes: list[str | None] = []
    actual_design: str | None = validated.design
    q_sorted = np.sort(q_all)
    for threshold in knots:
        z = (y <= threshold).astype(float)
        q = (q_audit <= threshold).astype(float)
        residual = z - q
        proxy_total = int(np.searchsorted(q_sorted, threshold, side="right"))
        corrected = (proxy_total + math.fsum(residual * inverse)) / N
        var_total = validated.variance_total(residual)
        note = validated.note
        raw.append(float(corrected))
        se.append(
            float(math.sqrt(max(0.0, var_total)) / N)
            if var_total is not None else math.nan
        )
        notes.append(note)
    projected = bounded_isotonic_cdf(np.asarray(raw))
    median_index = int(np.searchsorted(projected, quantile, side="left"))
    median_index = min(median_index, len(knots) - 1)
    census = len(audit_keys) == len(full_keys) and bool(np.all(inclusion == 1.0))
    note = next((x for x in notes if x), None)
    message = (
        "Exact finite-population quantile: all frame outcomes observed with pi=1."
        if census else
        "Bounded equal-knot isotonic projection of pointwise HT-corrected CDF; "
        "Scalar quantile SE unavailable; noncensus CIs are approximate."
    )
    if note:
        message += f" CDF design/provenance: {note}"
    frame = pd.DataFrame({
        "threshold": knots, "raw_cdf": raw,
        "admissible_cdf": projected, "cdf_se": se,
    })
    return GeneralizedDifferenceQuantileResult(
        statistic=statistic, estimate=float(knots[median_index]),
        quantile_level=quantile, population_size=N, audit_size=len(audit),
        inference="finite", variance_design="census" if census else actual_design,
        variance_note=message, projection_method="bounded_isotonic_equal_knots",
        _cdf=frame, standard_error=0.0 if census else None,
    )
