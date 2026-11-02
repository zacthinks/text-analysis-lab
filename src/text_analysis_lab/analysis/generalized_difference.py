"""Classical generalized-difference estimation for probability-audited measurements.

The estimator treats the focal artifact's surrogate values as auxiliary values
known for the full finite corpus, then uses inverse-probability weighted audit
residuals to correct their population total/mean.  No nuisance model is fitted.

TeAL probability sampling records permit design-based standard errors for
SRSWOR, stratified SRSWOR and independent Bernoulli audits. An external audit
can also explicitly declare one of these designs (and provide strata when
needed); its actual randomized selection is a researcher assertion, not
something TeAL can infer from first-order inclusion probabilities alone.
Design variance is conditional on the finite frame and frozen surrogate values.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import NormalDist
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from text_analysis_lab.core.errors import ArtifactError

if TYPE_CHECKING:
    from text_analysis_lab.core.artifact_base import BaseArtifact


@dataclass(frozen=True)
class GeneralizedDifferenceResult:
    """Ephemeral generalized-difference estimate over one finite corpus."""

    population_size: int
    audit_size: int
    surrogate_field: str
    gold_field: str
    pi_field: str
    surrogate_total: float
    surrogate_mean: float
    correction_total: float
    correction_mean: float
    estimated_total: float
    estimated_mean: float
    standard_error_total: float | None
    standard_error: float | None
    variance_design: str | None
    variance_note: str | None = None
    gold_is_binary: bool = False
    inference: str = "finite"
    variance_audit: float | None = None
    variance_corpus: float | None = None
    variance_total: float | None = None

    @property
    def estimated_prevalence(self) -> float:
        """Return the corrected mean when the audited gold values are binary."""
        if not self.gold_is_binary:
            raise ValueError(
                "estimated_prevalence is only defined for binary 0/1 gold values."
            )
        return self.estimated_mean

    def confidence_interval(
        self,
        *,
        level: float = 0.95,
        scale: str = "mean",
    ) -> tuple[float, float]:
        """Return a normal-approximation design interval for mean or total.

        The interval is intentionally not clipped to the support of the outcome;
        generalized-difference estimators and their normal intervals need not be
        bounded by 0 and 1 even when the gold variable is binary.
        """
        level = float(level)
        if not (0.0 < level < 1.0):
            raise ValueError("level must satisfy 0 < level < 1.")
        if scale == "mean":
            estimate = self.estimated_mean
            se = self.standard_error
        elif scale == "total":
            estimate = self.estimated_total
            se = self.standard_error_total
        else:
            raise ValueError("scale must be 'mean' or 'total'.")
        if se is None:
            note = self.variance_note or "design variance is unavailable."
            raise ValueError(f"Cannot form a confidence interval because {note}")
        z = NormalDist().inv_cdf(0.5 + level / 2.0)
        return (float(estimate - z * se), float(estimate + z * se))

    def p_value(self, *, null: float = 0.0, scale: str = "mean") -> float:
        """Return a two-sided normal-approximation design p-value.

        In finite mode this uses design uncertainty conditional on the
        realized corpus. In superpopulation mode it uses the combined corpus
        and audit variance, assuming IID corpus generation and frozen predictions.
        """
        if scale == "mean":
            estimate = self.estimated_mean
            se = self.standard_error
        elif scale == "total":
            estimate = self.estimated_total
            se = self.standard_error_total
        else:
            raise ValueError("scale must be 'mean' or 'total'.")
        if se is None:
            note = self.variance_note or "design variance is unavailable."
            raise ValueError(f"Cannot form a p-value because {note}")
        if se == 0.0:
            return 1.0 if float(estimate) == float(null) else 0.0
        z = abs((float(estimate) - float(null)) / float(se))
        return float(2.0 * (1.0 - NormalDist().cdf(z)))

    def to_frame(self) -> pd.DataFrame:
        """Return a tidy one-row summary suitable for notebook display."""
        row: dict[str, Any] = {
            "population_size": self.population_size,
            "audit_size": self.audit_size,
            "surrogate_mean": self.surrogate_mean,
            "correction_mean": self.correction_mean,
            "estimated_mean": self.estimated_mean,
            "standard_error": self.standard_error,
            "surrogate_total": self.surrogate_total,
            "correction_total": self.correction_total,
            "estimated_total": self.estimated_total,
            "standard_error_total": self.standard_error_total,
            "variance_design": self.variance_design,
        }
        if self.inference == "superpopulation":
            row.update({
                "inference": self.inference,
                "variance_audit": self.variance_audit,
                "variance_corpus": self.variance_corpus,
                "variance_total": self.variance_total,
            })
        if self.gold_is_binary:
            row["estimated_prevalence"] = self.estimated_mean
        return pd.DataFrame([row])


@dataclass(frozen=True)
class GeneralizedDifferenceContrastResult:
    """Difference between two generalized-difference subgroup means."""

    first: Any
    second: Any
    first_estimate: float
    second_estimate: float
    difference: float
    standard_error: float | None
    variance_design: str | None
    variance_note: str | None = None

    def confidence_interval(self, *, level: float = 0.95) -> tuple[float, float]:
        level = float(level)
        if not (0.0 < level < 1.0):
            raise ValueError("level must satisfy 0 < level < 1.")
        if self.standard_error is None:
            note = self.variance_note or "design variance is unavailable."
            raise ValueError(f"Cannot form a confidence interval because {note}")
        z = NormalDist().inv_cdf(0.5 + level / 2.0)
        return (
            float(self.difference - z * self.standard_error),
            float(self.difference + z * self.standard_error),
        )

    def p_value(self, *, null: float = 0.0) -> float:
        if self.standard_error is None:
            note = self.variance_note or "design variance is unavailable."
            raise ValueError(f"Cannot form a p-value because {note}")
        if self.standard_error == 0.0:
            return 1.0 if self.difference == float(null) else 0.0
        z = abs((self.difference - float(null)) / self.standard_error)
        return float(2.0 * (1.0 - NormalDist().cdf(z)))

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "first": self.first,
                    "second": self.second,
                    "first_estimate": self.first_estimate,
                    "second_estimate": self.second_estimate,
                    "difference": self.difference,
                    "standard_error": self.standard_error,
                    "p_value": None if self.standard_error is None else self.p_value(),
                    "variance_design": self.variance_design,
                }
            ]
        )


@dataclass(frozen=True)
class GeneralizedDifferenceGroupedResult:
    """Generalized-difference estimates for known finite-corpus subgroups."""

    group_field: str
    estimates: pd.DataFrame
    _contrast_inputs: Mapping[str, Any]

    def to_frame(self) -> pd.DataFrame:
        return self.estimates.copy()

    def contrast(self, first: Any, second: Any) -> GeneralizedDifferenceContrastResult:
        groups = self.estimates.set_index("group", drop=False)
        if first not in groups.index:
            raise KeyError(f"Unknown first group {first!r}.")
        if second not in groups.index:
            raise KeyError(f"Unknown second group {second!r}.")
        first_row = groups.loc[first]
        second_row = groups.loc[second]
        if isinstance(first_row, pd.DataFrame) or isinstance(second_row, pd.DataFrame):
            raise ArtifactError("Group labels must identify unique subgroup estimates.")
        residual = np.asarray(self._contrast_inputs["residual"], dtype=float)
        audit_groups = np.asarray(self._contrast_inputs["audit_groups"], dtype=object)
        N_first = int(first_row["population_size"])
        N_second = int(second_row["population_size"])
        transformed = residual * (
            (audit_groups == first).astype(float) / float(N_first)
            - (audit_groups == second).astype(float) / float(N_second)
        )
        variance, design, note = _design_variance_from_pi_provenance(
            surrogate=self._contrast_inputs["surrogate"],
            pi_artifact=self._contrast_inputs["pi_artifact"],
            audit_frame=self._contrast_inputs["audit_keys"].copy(),
            residual=transformed,
            population_keys=self._contrast_inputs["population_keys"],
            primary_key=self._contrast_inputs["primary_key"],
            inclusion=self._contrast_inputs["inclusion"],
            design=self._contrast_inputs["design"],
            strata=self._contrast_inputs["strata"],
        )
        se = None if variance is None else float(math.sqrt(max(variance, 0.0)))
        difference = float(first_row["estimated_mean"] - second_row["estimated_mean"])
        return GeneralizedDifferenceContrastResult(
            first=first,
            second=second,
            first_estimate=float(first_row["estimated_mean"]),
            second_estimate=float(second_row["estimated_mean"]),
            difference=difference,
            standard_error=se,
            variance_design=design,
            variance_note=note,
        )

    def contrasts(self, reference: Any) -> pd.DataFrame:
        """Return every non-reference group contrast against ``reference``."""
        groups = self.estimates["group"].tolist()
        if reference not in groups:
            raise KeyError(f"Unknown reference group {reference!r}.")
        frames = [
            self.contrast(group, reference).to_frame()
            for group in groups
            if group != reference
        ]
        if not frames:
            return pd.DataFrame(
                columns=[
                    "first",
                    "second",
                    "first_estimate",
                    "second_estimate",
                    "difference",
                    "standard_error",
                    "p_value",
                    "variance_design",
                ]
            )
        return pd.concat(frames, ignore_index=True)

    def pairwise_contrasts(self) -> pd.DataFrame:
        """Return all unordered pairwise group contrasts with design covariance."""
        groups = self.estimates["group"].tolist()
        frames = [
            self.contrast(groups[i], groups[j]).to_frame()
            for i in range(len(groups))
            for j in range(i + 1, len(groups))
        ]
        if not frames:
            return pd.DataFrame(
                columns=[
                    "first",
                    "second",
                    "first_estimate",
                    "second_estimate",
                    "difference",
                    "standard_error",
                    "p_value",
                    "variance_design",
                ]
            )
        return pd.concat(frames, ignore_index=True)


def generalized_difference_by(
    surrogate: BaseArtifact,
    *,
    gold: BaseArtifact | str,
    pi: BaseArtifact | str,
    group: BaseArtifact | str | None = None,
    group_field: str,
    surrogate_field: str = "prediction",
    gold_field: str = "label",
    pi_field: str = "pi",
    design: str | None = None,
    strata: BaseArtifact | str | None = None,
    inference: str = "finite",
    known: BaseArtifact | str | Sequence[BaseArtifact | str] | None = None,
) -> GeneralizedDifferenceGroupedResult:
    """Estimate corrected finite-corpus means/prevalences within known groups.

    ``group_field`` may be a data or inherited-metadata field. By default it is
    resolved from the surrogate artifact; pass ``group=`` when the grouping
    variable lives on another same-key artifact. ``result.contrast(a, b)``
    returns the corrected difference in group means with its design-based SE.
    """
    if known is not None and inference != "finite":
        raise ValueError("known= supports only finite-population inference.")
    if inference != "finite":
        raise ValueError("Grouped superpopulation inference is not supported; use inference='finite'.")
    from text_analysis_lab.analysis.gd_design_variance import check_design
    check_design(design)
    project = surrogate.project
    strata_artifact = None if strata is None else project.get_artifact(strata)
    gold_artifact = project.get_artifact(gold)
    pi_artifact = project.get_artifact(pi)
    group_artifact = surrogate if group is None else project.get_artifact(group)
    keys = tuple(str(value) for value in surrogate.primary_key)
    if not keys:
        raise ArtifactError("Surrogate artifact has no primary key.")
    for role, artifact in (
        ("gold", gold_artifact),
        ("pi", pi_artifact),
        ("group", group_artifact),
    ):
        if tuple(str(value) for value in artifact.primary_key) != keys:
            raise ArtifactError(
                f"Surrogate and {role} artifacts must use the same primary-key columns."
            )
    _require_field(surrogate, surrogate_field, role="surrogate")
    _require_field(gold_artifact, gold_field, role="gold")
    _require_field(pi_artifact, pi_field, role="pi")

    population = _frame(surrogate, surrogate_field)
    audit = _frame(gold_artifact, gold_field)
    pi_frame = _frame(pi_artifact, pi_field)
    group_frame = _field_frame(group_artifact, group_field, role="group")
    for role, frame in (
        ("surrogate", population),
        ("gold", audit),
        ("pi", pi_frame),
        ("group", group_frame),
    ):
        _validate_unique(frame, keys, role=role)
    population_keys = set(_key_tuples(population, keys))
    from text_analysis_lab.analysis.gd_known import known_table
    known_data, known_keys = known_table(
        project, known, primary_key=keys, label_field=gold_field,
    )
    if known is not None:
        from text_analysis_lab.analysis.estimation import _recorded_frame
        recorded = _recorded_frame(project, pi_artifact)
        if recorded is not None:
            if tuple(str(k) for k in recorded.primary_key) != keys:
                raise ArtifactError("Recorded audit frame has different primary keys.")
            recorded_table = _key_only_frame(recorded)
            _validate_unique(recorded_table, keys, role="recorded audit frame")
            remainder_keys = set(_key_tuples(recorded_table, keys))
            if known_keys & remainder_keys:
                raise ArtifactError("known= overlaps the recorded audit sampling frame.")
            if population_keys not in (remainder_keys, remainder_keys | known_keys):
                raise ArtifactError(
                    "Surrogate must cover the audit remainder U or U union known=."
                )
            variance_surrogate = recorded
        else:
            if not known_keys.issubset(population_keys):
                raise ArtifactError(
                    "External known= requires surrogate coverage of known keys."
                )
            remainder_keys = population_keys - known_keys
            variance_surrogate = surrogate
        if not remainder_keys:
            raise ArtifactError("No remaining probability-audit frame U.")
    else:
        remainder_keys = population_keys
        variance_surrogate = surrogate
    target_keys = remainder_keys | known_keys
    if set(_key_tuples(group_frame, keys)) != target_keys:
        raise ArtifactError(
            "group= must describe exactly the full target F, including known= keys."
        )
    audit_keys = set(_key_tuples(audit, keys))
    if set(_key_tuples(pi_frame, keys)) != audit_keys:
        raise ArtifactError("pi key set must exactly equal the gold/audit key set.")
    if audit_keys & known_keys:
        raise ArtifactError("known= overlaps audited gold labels.")
    if not audit_keys.issubset(remainder_keys):
        raise ArtifactError(
            "gold/audit contains keys outside the probability-audit remainder."
        )

    population = population.loc[
        [k in remainder_keys for k in _key_tuples(population, keys)]
    ].copy()
    pop = population.merge(
        group_frame, on=list(keys), validate="one_to_one", sort=False
    )
    known_groups = known_data.merge(
        group_frame, on=list(keys), validate="one_to_one", sort=False,
    )
    merged = (
        audit.merge(
            population.loc[:, [*keys, surrogate_field]],
            on=list(keys),
            validate="one_to_one",
            sort=False,
        )
        .merge(pi_frame, on=list(keys), validate="one_to_one", sort=False)
        .merge(group_frame, on=list(keys), validate="one_to_one", sort=False)
    )
    q_all = _numeric(pop[surrogate_field], name=f"surrogate field {surrogate_field!r}")
    y = _numeric(merged[gold_field], name=f"gold field {gold_field!r}")
    q_audit = _numeric(
        merged[surrogate_field], name=f"surrogate field {surrogate_field!r}"
    )
    inclusion = _numeric(merged[pi_field], name=f"pi field {pi_field!r}")
    if np.any(inclusion <= 0.0) or np.any(inclusion > 1.0):
        raise ArtifactError("pi values must satisfy 0 < pi <= 1.")
    residual = y - q_audit
    if pop[group_field].isna().any() or merged[group_field].isna().any():
        raise ArtifactError("group_field may not contain missing values.")

    rows: list[dict[str, Any]] = []
    audit_groups = merged[group_field].to_numpy(dtype=object)
    pop_groups = pop[group_field].to_numpy(dtype=object)
    group_values = list(pd.unique(group_frame[group_field]))
    if group_frame[group_field].isna().any():
        raise ArtifactError("group_field may not contain missing values.")
    for value in group_values:
        pop_mask = pop_groups == value
        audit_mask = audit_groups == value
        N_g = int(np.sum((group_frame[group_field] == value).to_numpy()))
        n_g = int(np.sum(audit_mask))
        surrogate_total = float(np.sum(q_all[pop_mask]))
        correction_total = float(
            np.sum((residual * audit_mask.astype(float)) / inclusion)
        )
        if known is not None:
            correction_total += float(known_groups.loc[
                known_groups[group_field] == value, "__known_y__"
            ].sum())
        estimated_total = surrogate_total + correction_total
        transformed = residual * audit_mask.astype(float) / float(N_g)
        variance_mean, variance_design, note = _design_variance_from_pi_provenance(
            surrogate=variance_surrogate,
            pi_artifact=pi_artifact,
            audit_frame=merged.loc[:, list(keys)].copy(),
            residual=transformed,
            population_keys=remainder_keys,
            primary_key=keys,
            inclusion=inclusion,
            design=design,
            strata=strata_artifact,
        )
        se_mean = (
            None if variance_mean is None else float(math.sqrt(max(variance_mean, 0.0)))
        )
        rows.append(
            {
                "group": value,
                "population_size": N_g,
                "audit_size": n_g,
                "surrogate_mean": surrogate_total / N_g,
                "correction_mean": correction_total / N_g,
                "estimated_mean": estimated_total / N_g,
                "standard_error": se_mean,
                "estimated_total": estimated_total,
                "variance_design": variance_design,
                "variance_note": note,
                "gold_is_binary": bool(np.all(np.isin(y, [0.0, 1.0]))),
            }
        )
    estimates = pd.DataFrame(rows)
    if not estimates.empty and bool(estimates["gold_is_binary"].all()):
        estimates["estimated_prevalence"] = estimates["estimated_mean"]
    return GeneralizedDifferenceGroupedResult(
        group_field=group_field,
        estimates=estimates,
        _contrast_inputs={
            "residual": residual,
            "audit_groups": audit_groups,
            "surrogate": variance_surrogate,
            "pi_artifact": pi_artifact,
            "audit_keys": merged.loc[:, list(keys)].copy(),
            "population_keys": remainder_keys,
            "primary_key": keys,
            "inclusion": inclusion,
            "design": design,
            "strata": strata_artifact,
        },
    )


def generalized_difference(
    surrogate: BaseArtifact,
    *,
    gold: BaseArtifact | str,
    pi: BaseArtifact | str,
    surrogate_field: str = "prediction",
    gold_field: str = "label",
    pi_field: str = "pi",
    design: str | None = None,
    strata: BaseArtifact | str | None = None,
    inference: str = "finite",
    known: BaseArtifact | str | Sequence[BaseArtifact | str] | None = None,
) -> GeneralizedDifferenceResult:
    """Correct a full-corpus surrogate using a probability audit.

    For finite population ``F`` and probability audit ``A``, this computes

    ``sum_F q_i + sum_A (y_i - q_i) / pi_i``

    for the population total and divides by ``|F|`` for the population mean.
    ``surrogate`` therefore defines the target finite population and must cover
    every audited key.  ``gold`` and ``pi`` must have exactly the same key set.

    No model is learned. Recorded TeAL probability sampling provenance can
    identify SRSWOR, stratified SRSWOR or independent Bernoulli variance.
    An external audit may pass ``design=`` (and full-frame ``strata=`` for
    stratification), subject to observable validation; actual randomization
    and frozen surrogate predictions remain the researcher's assumptions.
    Without verified provenance or an asserted design, the point estimate is
    returned with no estimated design standard error.
    """
    if known is not None and inference != "finite":
        raise ValueError("known= supports only finite-population inference.")
    if inference not in {"finite", "superpopulation"}:
        raise ValueError("inference must be 'finite' or 'superpopulation'.")
    from text_analysis_lab.analysis.gd_design_variance import check_design
    check_design(design)
    project = surrogate.project
    strata_artifact = None if strata is None else project.get_artifact(strata)
    gold_artifact = project.get_artifact(gold)
    pi_artifact = project.get_artifact(pi)

    _require_field(surrogate, surrogate_field, role="surrogate")
    _require_field(gold_artifact, gold_field, role="gold")
    _require_field(pi_artifact, pi_field, role="pi")

    keys = tuple(str(value) for value in surrogate.primary_key)
    if not keys:
        raise ArtifactError("Surrogate artifact has no primary key.")
    for role, artifact in (("gold", gold_artifact), ("pi", pi_artifact)):
        if tuple(str(value) for value in artifact.primary_key) != keys:
            raise ArtifactError(
                f"Surrogate and {role} artifacts must use the same primary-key columns."
            )

    population = _frame(surrogate, surrogate_field)
    audit = _frame(gold_artifact, gold_field)
    pi_frame = _frame(pi_artifact, pi_field)
    _validate_unique(population, keys, role="surrogate")
    _validate_unique(audit, keys, role="gold")
    _validate_unique(pi_frame, keys, role="pi")
    if len(population) == 0:
        raise ArtifactError(
            "Generalized-difference estimation requires a non-empty population."
        )
    if len(audit) == 0:
        raise ArtifactError(
            "Generalized-difference estimation requires a non-empty audit."
        )

    audit_keys = set(_key_tuples(audit, keys))
    pi_keys = set(_key_tuples(pi_frame, keys))
    population_keys = set(_key_tuples(population, keys))
    if pi_keys != audit_keys:
        raise ArtifactError("pi key set must exactly equal the gold/audit key set.")
    if not audit_keys.issubset(population_keys):
        missing = len(audit_keys - population_keys)
        raise ArtifactError(f"Surrogate is missing {missing} gold/audit key(s).")

    from text_analysis_lab.analysis.gd_known import known_table
    known_data, known_keys = known_table(
        project, known, primary_key=keys, label_field=gold_field,
    )
    if known_keys & audit_keys:
        raise ArtifactError("known= overlaps audited gold labels.")
    if known is not None:
        from text_analysis_lab.analysis.estimation import _recorded_frame
        recorded = _recorded_frame(project, pi_artifact)
        if recorded is not None:
            if tuple(str(k) for k in recorded.primary_key) != keys:
                raise ArtifactError("Recorded audit frame has different primary keys.")
            recorded_frame = _key_only_frame(recorded)
            _validate_unique(recorded_frame, keys, role="recorded audit frame")
            remainder_keys = set(_key_tuples(recorded_frame, keys))
            if known_keys & remainder_keys:
                raise ArtifactError("known= overlaps the recorded audit sampling frame.")
            if population_keys not in (remainder_keys, remainder_keys | known_keys):
                raise ArtifactError(
                    "Surrogate must cover the audit remainder U or U union known=."
                )
            variance_surrogate = recorded
        else:
            if not known_keys.issubset(population_keys):
                raise ArtifactError(
                    "External known= requires the surrogate to cover known keys."
                )
            remainder_keys = population_keys - known_keys
            variance_surrogate = surrogate
        if not remainder_keys or not audit_keys.issubset(remainder_keys):
            raise ArtifactError("Audit keys must lie in the remaining frame U.")
    else:
        remainder_keys = population_keys
        variance_surrogate = surrogate
    q_all = _numeric(
        population[surrogate_field], name=f"surrogate field {surrogate_field!r}"
    )
    merged = audit.merge(
        population.loc[:, [*keys, surrogate_field]],
        on=list(keys),
        how="left",
        validate="one_to_one",
        sort=False,
    ).merge(
        pi_frame,
        on=list(keys),
        how="left",
        validate="one_to_one",
        sort=False,
    )
    y = _numeric(merged[gold_field], name=f"gold field {gold_field!r}")
    q_audit = _numeric(
        merged[surrogate_field], name=f"surrogate field {surrogate_field!r}"
    )
    inclusion = _numeric(merged[pi_field], name=f"pi field {pi_field!r}")
    if np.any(inclusion <= 0.0) or np.any(inclusion > 1.0):
        raise ArtifactError("pi values must satisfy 0 < pi <= 1.")

    residual = y - q_audit
    N = len(remainder_keys) + len(known_keys)
    if known is not None:
        # The exactly known K labels replace, rather than augment, q on K.
        is_remaining = np.asarray([
            key in remainder_keys for key in _key_tuples(population, keys)
        ], dtype=bool)
        surrogate_total = float(q_all[is_remaining].sum())
    else:
        surrogate_total = float(q_all.sum())
    correction_total = float(np.sum(residual / inclusion))
    if known is not None:
        correction_total += float(known_data["__known_y__"].sum())
    estimated_total = surrogate_total + correction_total

    variance_total, design, variance_note = _design_variance_from_pi_provenance(
        surrogate=variance_surrogate,
        pi_artifact=pi_artifact,
        audit_frame=merged.loc[:, [*keys]].copy(),
        residual=residual,
        population_keys=remainder_keys,
        primary_key=keys,
        inclusion=inclusion,
        design=design,
        strata=strata_artifact,
    )
    variance_audit = None if variance_total is None else float(variance_total / N**2)
    variance_corpus = None
    combined_variance = variance_audit
    if inference == "superpopulation":
        if variance_audit is None or design is None:
            raise ValueError(
                "Superpopulation inference requires a verified or asserted "
                "supported audit design and an estimable audit variance."
            )
        from text_analysis_lab.analysis.gd_superpopulation import corpus_variance
        h_full, h_sample = None, None
        if design == "stratified_srswor":
            strata_for_model = strata_artifact
            if strata_for_model is None:
                op = project.get_operation(str(pi_artifact.operation_id))
                recorded_sources = project.operation_sources(str(pi_artifact.operation_id))
                ids = {
                    str(row["source_label"]): str(row["source_artifact_id"])
                    for row in recorded_sources
                }
                if "strata" not in ids:
                    raise ValueError("Stratified superpopulation inference requires strata.")
                strata_for_model = project.get_artifact(ids["strata"])
            fields = [str(col) for col in strata_for_model.get_data_columns()]
            if len(fields) != 1:
                raise ArtifactError("Superpopulation strata must have one data field.")
            frame = _frame(strata_for_model, fields[0])
            _validate_unique(frame, keys, role="superpopulation strata")
            if set(_key_tuples(frame, keys)) != population_keys:
                raise ArtifactError("Superpopulation strata must cover the frame.")
            population_aligned = population.loc[:, list(keys)].merge(
                frame, on=list(keys), validate="one_to_one", sort=False
            )
            audit_aligned = merged.loc[:, list(keys)].merge(
                frame, on=list(keys), validate="one_to_one", sort=False
            )
            h_full = population_aligned[fields[0]].to_numpy()
            h_sample = audit_aligned[fields[0]].to_numpy()
        variance_corpus = corpus_variance(
            gold=y, inclusion=inclusion, population_size=N,
            design=design, stratum_population=h_full, stratum_audit=h_sample,
        )
        combined_variance = variance_audit + variance_corpus
    se_mean = (
        None if combined_variance is None
        else float(math.sqrt(max(combined_variance, 0.0)))
    )
    se_total = None if se_mean is None else float(se_mean * N)

    return GeneralizedDifferenceResult(
        population_size=N,
        audit_size=len(audit),
        surrogate_field=surrogate_field,
        gold_field=gold_field,
        pi_field=pi_field,
        surrogate_total=surrogate_total,
        surrogate_mean=float(surrogate_total / N),
        correction_total=correction_total,
        correction_mean=float(correction_total / N),
        estimated_total=float(estimated_total),
        estimated_mean=float(estimated_total / N),
        standard_error_total=se_total,
        standard_error=se_mean,
        variance_design=design,
        variance_note=variance_note,
        gold_is_binary=bool(np.all(np.isin(y, [0.0, 1.0]))),
        inference=inference,
        variance_audit=variance_audit,
        variance_corpus=variance_corpus,
        variance_total=combined_variance,
    )


def _design_variance_from_pi_provenance(
    *, surrogate, pi_artifact, audit_frame, residual, population_keys,
    primary_key, inclusion, design=None, strata=None,
) -> tuple[float | None, str | None, str | None]:
    from text_analysis_lab.analysis.gd_design_variance import design_variance
    return design_variance(
        surrogate=surrogate, pi_artifact=pi_artifact,
        audit_frame=audit_frame, residual=residual,
        population_keys=population_keys, primary_key=primary_key,
        inclusion=inclusion, design=design, strata=strata,
        srs_variance=_srswor_variance,
        stratified_variance=_stratified_srswor_variance,
        unavailable=_variance_unavailable,
        key_frame=_key_only_frame, field_frame=_frame,
        validate_unique=_validate_unique, key_tuples=_key_tuples,
    )


def _srswor_variance(residual: np.ndarray, *, N: int, n: int) -> float | None:
    """Estimated variance of an HT residual total under SRS without replacement."""
    if n <= 0 or N <= 0 or n > N:
        raise ValueError("SRS variance requires 0 < n <= N.")
    if n == N:
        return 0.0
    if n < 2:
        return None
    s2 = float(np.var(np.asarray(residual, dtype=float), ddof=1))
    f = float(n) / float(N)
    return float((N**2) * (1.0 - f) * s2 / n)


def _stratified_srswor_variance(
    *,
    strata_frame: pd.DataFrame,
    audit_residuals: pd.DataFrame,
    primary_key: tuple[str, ...],
    stratum_field: str,
) -> tuple[float | None, str | None, str | None]:
    """Estimated HT residual-total variance for fixed-size SRSWOR within strata."""
    merged = audit_residuals.merge(
        strata_frame.loc[:, [*primary_key, stratum_field]],
        on=list(primary_key),
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if merged[stratum_field].isna().any():
        return _variance_unavailable(
            "some audited keys are missing a sampling stratum."
        )

    population_counts = (
        strata_frame.groupby(stratum_field, dropna=False).size().to_dict()
    )
    audit_counts = merged.groupby(stratum_field, dropna=False).size().to_dict()
    variance = 0.0
    for stratum, N_h_value in population_counts.items():
        N_h = int(N_h_value)
        n_h = int(audit_counts.get(stratum, 0))
        if n_h <= 0:
            return _variance_unavailable(
                f"sampling stratum {stratum!r} has no audited observations, so its residual total "
                "cannot be design-corrected."
            )
        if n_h > N_h:
            return _variance_unavailable(
                f"sampling stratum {stratum!r} has n_h={n_h} greater than N_h={N_h}."
            )
        if n_h == N_h:
            continue
        if n_h < 2:
            return _variance_unavailable(
                f"sampling stratum {stratum!r} has only n_h={n_h} audited observation; at least "
                "two are needed to estimate its within-stratum residual variance."
            )
        values = merged.loc[merged[stratum_field] == stratum, "__residual__"].to_numpy(
            dtype=float
        )
        s2_h = float(np.var(values, ddof=1))
        f_h = float(n_h) / float(N_h)
        variance += float((N_h**2) * (1.0 - f_h) * s2_h / n_h)
    return float(variance), "stratified_srswor", None


def _variance_unavailable(note: str) -> tuple[None, None, str]:
    return None, None, str(note)


def _frame(artifact: BaseArtifact, field: str) -> pd.DataFrame:
    value = artifact.query(
        key_columns=True,
        data_columns=[field],
        metadata_columns=False,
        form="table",
        include_position=False,
    )
    if not isinstance(value, pd.DataFrame):
        raise ArtifactError(f"Could not materialize {field!r} as a table.")
    return value.reset_index(drop=True)


def _field_frame(artifact: BaseArtifact, field: str, *, role: str) -> pd.DataFrame:
    data_columns = [str(value) for value in artifact.get_data_columns()]
    if field in data_columns:
        return _frame(artifact, field)
    metadata_columns = [str(value) for value in artifact.get_full_metadata_columns()]
    if field not in metadata_columns:
        raise ArtifactError(
            f"{role} artifact does not expose field {field!r} as data or inherited metadata; "
            f"data={data_columns}, metadata={metadata_columns}."
        )
    value = artifact.query(
        key_columns=True,
        data_columns=False,
        metadata_columns=[field],
        metadata_mode="full",
        form="table",
        include_position=False,
    )
    if not isinstance(value, pd.DataFrame):
        raise ArtifactError(f"Could not materialize {role} field {field!r} as a table.")
    return value.loc[:, [*artifact.primary_key, field]].reset_index(drop=True)


def _key_only_frame(artifact: BaseArtifact) -> pd.DataFrame:
    value = artifact.query(
        key_columns=True,
        data_columns=False,
        metadata_columns=False,
        form="table",
        include_position=False,
    )
    if not isinstance(value, pd.DataFrame):
        raise ArtifactError("Could not materialize artifact keys as a table.")
    keys = [str(value) for value in artifact.primary_key]
    return value.loc[:, keys].reset_index(drop=True)


def _require_field(artifact: BaseArtifact, field: str, *, role: str) -> None:
    columns = [str(value) for value in artifact.get_data_columns()]
    if field not in columns:
        raise ArtifactError(
            f"{role} artifact must expose data field {field!r}; available fields are {columns}."
        )


def _validate_unique(frame: pd.DataFrame, keys: Sequence[str], *, role: str) -> None:
    missing = [key for key in keys if key not in frame.columns]
    if missing:
        raise ArtifactError(f"{role} frame is missing primary-key column(s) {missing}.")
    if frame.duplicated(subset=list(keys)).any():
        raise ArtifactError(f"{role} frame contains duplicate primary keys.")


def _key_tuples(frame: pd.DataFrame, keys: Sequence[str]) -> list[tuple[int, ...]]:
    return [
        tuple(int(value) for value in row)
        for row in frame.loc[:, list(keys)].itertuples(index=False, name=None)
    ]


def _numeric(series: pd.Series, *, name: str) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    if np.any(~np.isfinite(values)):
        raise ArtifactError(f"{name} values must all be finite numeric values.")
    return values
