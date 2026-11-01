"""Direct design-based inference from high-warrant human measurements.

Only use this estimator when the audited outcome is a defensible human-coded
measurement under a documented, frozen coding procedure. This is a scientific
input contract, not a claim TeAL can establish warrant from numeric values or
field names. Do not substitute uncorrected surrogate/model predictions; use
generalized_difference() with an independent high-warrant audit instead.

The target finite frame must be provided or recovered from verified TeAL
probability-sampling provenance. An external design assertion cannot invent N.
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
    """Ephemeral HT mean/proportion with explicit inference and variance scale."""

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
            raise ValueError("estimated_prevalence requires binary 0/1 gold labels.")
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


def _gold_data_field(gold: BaseArtifact, requested: str | None) -> str:
    """Select the audited measurement field, never a key or metadata column.

    A single data field is unambiguous regardless of its name; multi-field
    artifacts require explicit selection rather than silently guessing which
    variable is the human measurement.
    """
    columns = [str(column) for column in gold.get_data_columns()]
    if requested is None:
        if len(columns) != 1:
            raise ArtifactError(
                "gold_field= is required unless the gold artifact has exactly "
                f"one data field; available data fields: {columns}."
            )
        return columns[0]
    _require_field(gold, requested, role="gold")
    return requested


def estimate(
    gold: BaseArtifact,
    *,
    pi: BaseArtifact | str,
    population: BaseArtifact | str | None = None,
    gold_field: str | None = None,
    pi_field: str = "pi",
    statistic: str = "mean",
    design: str | None = None,
    strata: BaseArtifact | str | None = None,
    inference: str = "finite",
) -> EstimateResult:
    """HT mean or proportion of a probability-audited, known finite frame.

    With TeAL-recorded pi, the frame and design can be recovered automatically.
    For external pi, provide the full keyed population frame and assert design=
    to obtain a conditional SE. An external assertion is not proof of sampling.
    `gold` must hold high-warrant human measurements, not model predictions.
    Numeric data, operator provenance and field names do not prove this:
    the researcher is responsible for the measurement's warrant and for keeping
    the audit independent of model selection and fitting. If the artifact has
    exactly one data field, gold_field is inferred; otherwise it is required.
    """
    if statistic != "mean":
        raise ValueError("Only statistic='mean' is implemented at this checkpoint.")
    if inference not in {"finite", "superpopulation"}:
        raise ValueError("inference must be 'finite' or 'superpopulation'.")
    check_design(design)
    project = gold.project
    pi_artifact = project.get_artifact(pi)
    frame_artifact = (
        _recorded_frame(project, pi_artifact)
        if population is None else project.get_artifact(population)
    )
    if frame_artifact is None:
        raise ValueError(
            "population= is required for external pi without verified TeAL "
            "sampling-frame provenance; inclusion probabilities cannot identify N."
        )
    strata_artifact = None if strata is None else project.get_artifact(strata)
    gold_field = _gold_data_field(gold, gold_field)
    _require_field(pi_artifact, pi_field, role="pi")
    keys = tuple(str(x) for x in gold.primary_key)
    if not keys:
        raise ArtifactError("Gold has no primary-key columns.")
    for role, art in (("pi", pi_artifact), ("population", frame_artifact)):
        if tuple(str(x) for x in art.primary_key) != keys:
            raise ArtifactError(f"{role} and gold primary-key columns differ.")
    frame = _key_only_frame(frame_artifact)
    audit = _frame(gold, gold_field)
    probabilities = _frame(pi_artifact, pi_field)
    for role, table in (
        ("population", frame), ("gold", audit), ("pi", probabilities)
    ):
        _validate_unique(table, keys, role=role)
    full_keys = set(_key_tuples(frame, keys))
    gold_keys = set(_key_tuples(audit, keys))
    pi_keys = set(_key_tuples(probabilities, keys))
    if not full_keys:
        raise ArtifactError("Direct estimation requires a non-empty population.")
    if not gold_keys:
        raise ArtifactError("Direct estimation requires at least one audited label.")
    if gold_keys != pi_keys:
        raise ArtifactError("Gold and pi must have identical audit key sets.")
    if not gold_keys.issubset(full_keys):
        raise ArtifactError("Gold contains keys outside the population frame.")
    merged = audit.merge(
        probabilities, on=list(keys), validate="one_to_one", sort=False
    )
    y = _numeric(merged[gold_field], name=f"gold field {gold_field!r}")
    inclusion = _numeric(merged[pi_field], name=f"pi field {pi_field!r}")
    if np.any((inclusion <= 0) | (inclusion > 1)):
        raise ArtifactError("pi must satisfy 0 < pi <= 1.")
    N = len(frame)
    ht_total = float(np.sum(y / inclusion))
    audit_variance_total, actual_design, note = _design_variance_from_pi_provenance(
        surrogate=frame_artifact, pi_artifact=pi_artifact,
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
        gold_is_binary=bool(np.all(np.isin(y, [0., 1.]))),
    )
