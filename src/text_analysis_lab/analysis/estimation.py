"""Descriptive and probability-sample estimation of numeric artifact fields.

An unweighted full-artifact mean is exact for those recorded values. It does
not establish that surrogate predictions measure the underlying human outcome.
Probability-audit inference requires an actual randomized sampling design;
scientific interpretation remains the researcher's responsibility.
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from text_analysis_lab.analysis.gd_design_variance import check_design
from text_analysis_lab.analysis.gd_superpopulation import corpus_variance
from text_analysis_lab.analysis.generalized_difference import (
    _design_variance_from_pi_provenance,
    _frame, _key_only_frame, _key_tuples, _numeric, _require_field,
    _validate_unique,
)
from text_analysis_lab.core.errors import ArtifactError

if TYPE_CHECKING:
    from text_analysis_lab.core.artifact_base import BaseArtifact


@dataclass(frozen=True)
class EstimateResult:
    """Ephemeral descriptive or HT mean/proportion with explicit variance scale."""

    statistic: str
    estimate: float
    estimated_total: float
    population_size: int
    audit_size: int
    standard_error: float | None
    standard_error_total: float | None
    inference: str
    variance_design: str | None
    variance_note: str | None
    variance_audit: float | None
    variance_corpus: float | None
    variance_total: float | None
    gold_is_binary: bool

    @property
    def estimated_mean(self) -> float:
        return self.estimate

    @property
    def estimated_prevalence(self) -> float:
        if not self.gold_is_binary:
            raise ValueError("estimated_prevalence requires binary 0/1 values.")
        return self.estimate

    def confidence_interval(
        self, *, level: float = 0.95, scale: str = "mean"
    ) -> tuple[float, float]:
        if not 0 < level < 1:
            raise ValueError("level must satisfy 0 < level < 1.")
        if scale == "mean":
            estimate, se = self.estimate, self.standard_error
        elif scale == "total":
            estimate, se = self.estimated_total, self.standard_error_total
        else:
            raise ValueError("scale must be 'mean' or 'total'.")
        if se is None:
            raise ValueError(
                "Cannot form an interval because "
                + (self.variance_note or "sampling variance is unavailable.")
            )
        z = NormalDist().inv_cdf(.5 + level / 2)
        return float(estimate - z * se), float(estimate + z * se)

    def p_value(self, *, null: float = 0.0, scale: str = "mean") -> float:
        if scale == "mean":
            value, se = self.estimate, self.standard_error
        elif scale == "total":
            value, se = self.estimated_total, self.standard_error_total
        else:
            raise ValueError("scale must be 'mean' or 'total'.")
        if se is None:
            raise ValueError(
                "Cannot test because "
                + (self.variance_note or "sampling variance is unavailable.")
            )
        if se == 0:
            return 1.0 if value == float(null) else 0.0
        return float(2 * (1 - NormalDist().cdf(abs((value - float(null)) / se))))

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "statistic": self.statistic,
            "estimate": self.estimate,
            "estimated_total": self.estimated_total,
            "population_size": self.population_size,
            "audit_size": self.audit_size,
            "standard_error": self.standard_error,
            "standard_error_total": self.standard_error_total,
            "inference": self.inference,
            "variance_design": self.variance_design,
            "variance_note": self.variance_note,
            "variance_audit": self.variance_audit,
            "variance_corpus": self.variance_corpus,
            "variance_total": self.variance_total,
        }])


def _recorded_frame(project, pi_artifact):
    """Recover the exact frame only from a verified TeAL sampling operation."""
    operation_id = getattr(pi_artifact, "operation_id", None)
    if operation_id is None:
        return None
    try:
        op = project.get_operation(str(operation_id))
        operator = project.get_operator(str(op["operator_id"]))
        from text_analysis_lab.core.probability_sample import ProbabilitySampleTranslator
        from text_analysis_lab.core.probability_split import ProbabilitySplitTranslator
        if not isinstance(operator, (ProbabilitySampleTranslator, ProbabilitySplitTranslator)):
            return None
        outputs = project.operation_outputs(str(operation_id))
        pis = [x for x in outputs if x.get("output_label") == "pi"]
        if len(pis) != 1 or str(pis[0]["artifact_id"]) != str(pi_artifact.artifact_id):
            return None
        source_rows = project.operation_sources(str(operation_id))
        sources = {
            str(x["source_label"]): str(x["source_artifact_id"])
            for x in source_rows
        }
        if "documents" not in sources:
            return None
        return project.get_artifact(sources["documents"])
    except (KeyError, ValueError, AttributeError, ArtifactError, FileNotFoundError):
        return None


def _data_field(artifact: BaseArtifact, requested: str | None) -> str:
    """Select a numeric data field, never a key or metadata column.

    A single data field is unambiguous regardless of its name; multi-field
    artifacts require explicit selection rather than silently guessing.
    """
    columns = [str(column) for column in artifact.get_data_columns()]
    if requested is None:
        if len(columns) != 1:
            raise ArtifactError(
                "field= is required unless the artifact has exactly "
                f"one data field; available data fields: {columns}."
            )
        return columns[0]
    _require_field(artifact, requested, role="input")
    return requested


def estimate(
    artifact: BaseArtifact,
    *,
    pi: BaseArtifact | str | None = None,
    population: BaseArtifact | str | None = None,
    field: str | None = None,
    pi_field: str = "pi",
    statistic: str = "mean",
    known: BaseArtifact | str | list[BaseArtifact | str] | None = None,
    design: str | None = None,
    strata: BaseArtifact | str | None = None,
    inference: str = "finite",
) -> EstimateResult:
    """Mean/proportion of a complete artifact or an HT probability sample.

    With no pi, compute the exact descriptive mean of recorded numeric values.
    This has no finite-population sampling uncertainty for those recorded values
    and conveys no guarantee about prediction accuracy or measurement warrant.
    With pi, a verified sampling frame or explicit population= is necessary.
    """
    if statistic != "mean":
        raise ValueError("Only statistic='mean' is implemented at this checkpoint.")
    if inference not in {"finite", "superpopulation"}:
        raise ValueError("inference must be 'finite' or 'superpopulation'.")
    if known is not None and inference != "finite":
        raise ValueError("known= supports only finite-population inference.")
    check_design(design)
    project = artifact.project
    field = _data_field(artifact, field)
    if pi is None:
        if known is not None:
            raise ValueError("known= requires a separate probability audit with pi=.")
        if population is not None or strata is not None or design is not None:
            raise ValueError("population=, strata= and design= require pi=.")
        values_frame = _frame(artifact, field)
        keys = tuple(str(x) for x in artifact.primary_key)
        _validate_unique(values_frame, keys, role="input")
        if len(values_frame) == 0:
            raise ArtifactError("Cannot estimate an empty artifact.")
        values = _numeric(values_frame[field], name=f"field {field!r}")
        if inference == "superpopulation":
            if len(values) < 2:
                raise ValueError("IID superpopulation variance requires at least two observations.")
            corpus = float(np.var(values, ddof=1) / len(values))
            se = float(np.sqrt(corpus))
        else:
            corpus = None
            se = 0.0
        total = float(np.sum(values))
        return EstimateResult(
            statistic=statistic, estimate=total / len(values),
            estimated_total=total, population_size=len(values),
            audit_size=0, standard_error=se,
            standard_error_total=float(se * len(values)),
            inference=inference, variance_design="census",
            variance_note="Exact descriptive mean of the recorded values; "
                "not an assertion about human-label accuracy or construct validity.",
            variance_audit=0.0, variance_corpus=corpus,
            variance_total=0.0 if corpus is None else corpus,
            gold_is_binary=bool(np.all(np.isin(values, [0., 1.]))),
        )
    pi_artifact = project.get_artifact(pi)
    # Resolve recorded provenance even when an explicit population is supplied.
    # Otherwise a different denominator can silently corrupt the HT mean
    # while the design-variance helper merely suppresses its SE.
    recorded_frame_artifact = _recorded_frame(project, pi_artifact)
    frame_artifact = (
        recorded_frame_artifact
        if population is None else project.get_artifact(population)
    )
    if frame_artifact is None:
        raise ValueError(
            "population= is required for external pi without verified TeAL "
            "sampling-frame provenance; inclusion probabilities cannot identify N."
        )
    strata_artifact = None if strata is None else project.get_artifact(strata)

    _require_field(pi_artifact, pi_field, role="pi")
    keys = tuple(str(x) for x in artifact.primary_key)
    if not keys:
        raise ArtifactError("Gold has no primary-key columns.")
    for role, art in (("pi", pi_artifact), ("population", frame_artifact)):
        if tuple(str(x) for x in art.primary_key) != keys:
            raise ArtifactError(f"{role} and input primary-key columns differ.")
    frame = _key_only_frame(frame_artifact)
    from text_analysis_lab.analysis.gd_known import known_table
    known_data, known_keys = known_table(
        project, known, primary_key=keys, label_field=field,
    )
    recorded_keys = None
    if recorded_frame_artifact is not None:
        recorded_frame = _key_only_frame(recorded_frame_artifact)
        _validate_unique(recorded_frame, keys, role="recorded sampling frame")
        recorded_keys = set(_key_tuples(recorded_frame, keys))
    if known is not None:
        supplied_keys = set(_key_tuples(frame, keys))
        if recorded_keys is not None:
            if known_keys & recorded_keys:
                raise ArtifactError("known= overlaps the recorded audit sampling frame.")
            if supplied_keys not in (recorded_keys, recorded_keys | known_keys):
                raise ArtifactError("population= must describe U or U union known=.")
            eligible_keys = recorded_keys
        else:
            if known_keys & supplied_keys and not known_keys.issubset(supplied_keys):
                raise ArtifactError(
                    "External population= must contain either all known= keys or none."
                )
            eligible_keys = supplied_keys - known_keys
        if not eligible_keys:
            raise ArtifactError("known= leaves no probability-audit remainder.")
    else:
        eligible_keys = set(_key_tuples(frame, keys))
    if known is None and population is not None and recorded_frame_artifact is not None:
        if tuple(str(k) for k in recorded_frame_artifact.primary_key) != keys:
            raise ArtifactError(
                "Explicit population= has different primary keys from the "
                "recorded sampling frame."
            )
        recorded_frame = _key_only_frame(recorded_frame_artifact)
        _validate_unique(recorded_frame, keys, role="recorded sampling frame")
        _validate_unique(frame, keys, role="explicit population")
        if set(_key_tuples(frame, keys)) != set(_key_tuples(recorded_frame, keys)):
            raise ArtifactError(
                "Explicit population= contradicts the recorded sampling frame; "
                "a full-frame estimate requires identical population keys."
            )
    audit = _frame(artifact, field)
    probabilities = _frame(pi_artifact, pi_field)
    for role, table in (
        ("population", frame), ("gold", audit), ("pi", probabilities)
    ):
        _validate_unique(table, keys, role=role)
    full_keys = eligible_keys
    gold_keys = set(_key_tuples(audit, keys))
    pi_keys = set(_key_tuples(probabilities, keys))
    if not full_keys:
        raise ArtifactError("Direct estimation requires a non-empty population.")
    if not gold_keys:
        raise ArtifactError("Direct estimation requires at least one audited label.")
    if gold_keys != pi_keys:
        raise ArtifactError("Gold and pi must have identical audit key sets.")
    if known_keys & gold_keys:
        raise ArtifactError("known= and probability-audit labels overlap.")
    if not gold_keys.issubset(full_keys):
        raise ArtifactError("Gold contains keys outside the population frame.")
    # The measurement field can legitimately be named "pi". Keep
    # inclusion probabilities in a distinct internal column through the merge.
    inclusion_column = "__teal_estimation_inclusion__"
    while inclusion_column in audit.columns or inclusion_column in probabilities.columns:
        inclusion_column += "_"
    merged = audit.merge(
        probabilities.rename(columns={pi_field: inclusion_column}),
        on=list(keys), validate="one_to_one", sort=False,
    )
    y = _numeric(merged[field], name=f"field {field!r}")
    inclusion = _numeric(
        merged[inclusion_column], name=f"pi field {pi_field!r}"
    )
    if np.any((inclusion <= 0) | (inclusion > 1)):
        raise ArtifactError("pi must satisfy 0 < pi <= 1.")
    N = len(full_keys) + len(known_keys)
    ht_total = float(np.sum(y / inclusion)) + float(known_data["__known_y__"].sum())
    audit_variance_total, actual_design, note = _design_variance_from_pi_provenance(
        surrogate=recorded_frame_artifact or frame_artifact, pi_artifact=pi_artifact,
        audit_frame=merged.loc[:, list(keys)].copy(),
        residual=y, population_keys=full_keys, primary_key=keys,
        inclusion=inclusion, design=design, strata=strata_artifact,
    )
    variance_audit = (
        None if audit_variance_total is None
        else float(audit_variance_total / N**2)
    )
    variance_corpus = None
    total_variance = variance_audit
    if inference == "superpopulation":
        if variance_audit is None or actual_design is None:
            raise ValueError(
                "Superpopulation inference requires a supported, verified "
                "or asserted design with estimable audit variance."
            )
        h_full = h_audit = None
        if actual_design == "stratified_srswor":
            if strata_artifact is None:
                op = project.get_operation(str(pi_artifact.operation_id))
                source_rows = project.operation_sources(str(pi_artifact.operation_id))
                sources = {str(row["source_label"]): str(row["source_artifact_id"])
                           for row in source_rows}
                if "strata" not in sources:
                    raise ValueError(
                        "Stratified superpopulation inference requires strata=."
                    )
                strata_artifact = project.get_artifact(sources["strata"])
            fields = [str(x) for x in strata_artifact.get_data_columns()]
            if len(fields) != 1:
                raise ArtifactError("Strata must have one data column.")
            h_frame = _frame(strata_artifact, fields[0])
            _validate_unique(h_frame, keys, role="strata")
            if set(_key_tuples(h_frame, keys)) != full_keys:
                raise ArtifactError("Strata must cover the population frame.")
            h_full = frame.merge(
                h_frame, on=list(keys), validate="one_to_one", sort=False
            )[fields[0]].to_numpy()
            h_audit = merged.loc[:, list(keys)].merge(
                h_frame, on=list(keys), validate="one_to_one", sort=False
            )[fields[0]].to_numpy()
        variance_corpus = corpus_variance(
            gold=y, inclusion=inclusion, population_size=N,
            design=actual_design,
            stratum_population=h_full, stratum_audit=h_audit,
        )
        total_variance = variance_audit + variance_corpus

    se = None if total_variance is None else float(
        np.sqrt(max(total_variance, 0.0))
    )
    return EstimateResult(
        statistic=statistic, estimate=ht_total / N,
        estimated_total=ht_total, population_size=N, audit_size=len(audit),
        standard_error=se,
        standard_error_total=None if se is None else float(N * se),
        inference=inference, variance_design=actual_design,
        variance_note=note, variance_audit=variance_audit,
        variance_corpus=variance_corpus, variance_total=total_variance,
        gold_is_binary=bool(
            np.all(np.isin(y, [0., 1.])) and
            np.all(np.isin(known_data["__known_y__"], [0., 1.]))
        ),
    )
