"""Recover supported survey designs and estimate conditional HT residual variance.

External design declarations assert actual randomization; marginal inclusion
probabilities cannot identify dependence between selection indicators.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from text_analysis_lab.core.errors import ArtifactError, ArtifactNotFoundError


DESIGNS = {"srswor", "stratified_srswor", "bernoulli"}


def check_design(design):
    if design is not None and design not in DESIGNS:
        raise ValueError(
            "design must be 'srswor', 'stratified_srswor', 'bernoulli', or None."
        )


def design_variance(
    *, surrogate, pi_artifact, audit_frame, residual, population_keys,
    primary_key, inclusion, design=None, strata=None,
    srs_variance, stratified_variance, unavailable, key_frame,
    field_frame, validate_unique, key_tuples,
):
    """Compute an HT residual-total variance using verified or asserted design."""
    check_design(design)
    project = surrogate.project
    N, n = len(population_keys), len(audit_frame)
    if n == 0:
        return unavailable("There is no audited observation.")
    internal, sources, outputs = None, [], []
    operation_id = getattr(pi_artifact, "operation_id", None)
    if operation_id is not None and all(hasattr(project, x) for x in (
        "get_operation", "get_operator", "operation_sources", "operation_outputs",
    )):
        try:
            operation = project.get_operation(str(operation_id))
            candidate = project.get_operator(str(operation["operator_id"]))
            from text_analysis_lab.core.probability_split import ProbabilitySplitTranslator
            from text_analysis_lab.core.probability_sample import ProbabilitySampleTranslator
            if isinstance(candidate, (ProbabilitySplitTranslator, ProbabilitySampleTranslator)):
                internal = candidate
                sources = project.operation_sources(str(operation_id))
                outputs = project.operation_outputs(str(operation_id))
        except Exception as exc:  # noqa: BLE001
            return unavailable(f"Sampling provenance could not be verified ({exc}).")

    by_source = {}
    if internal is not None:
        from text_analysis_lab.core.probability_split import ProbabilitySplitTranslator
        actual = (
            internal.design if hasattr(internal, "design") else
            "stratified_srswor" if internal.allocation is not None else "srswor"
        )
        if design is not None and design != actual:
            raise ArtifactError(
                f"Asserted design {design!r} contradicts TeAL sampling design {actual!r}."
            )
        design = actual
        pi_outputs = [o for o in outputs if o.get("output_label") == "pi"]
        if len(pi_outputs) != 1 or str(pi_outputs[0]["artifact_id"]) != str(
            pi_artifact.artifact_id
        ):
            return unavailable("pi is not the recorded sampling operation output.")
        by_source = {
            str(row["source_label"]): str(row["source_artifact_id"])
            for row in sources
        }
        try:
            original = key_frame(project.get_artifact(by_source["documents"]))
            validate_unique(original, primary_key, role="sampling frame")
        except Exception as exc:  # noqa: BLE001
            return unavailable(f"Sampling frame unavailable ({exc}).")
        if set(key_tuples(original, primary_key)) != population_keys:
            return unavailable("The surrogate population is not the sampled frame.")
        if getattr(internal, "n", None) is not None and internal.n != n:
            return unavailable("Audit size contradicts the recorded sample size.")
        sample_outputs = [
            o for o in outputs if o.get("output_label") == internal.sample_label
        ]
        if len(sample_outputs) != 1:
            return unavailable("Recorded sample output is missing.")
        try:
            sampled = key_frame(project.get_artifact(
                str(sample_outputs[0]["artifact_id"])
            ))
        except (ArtifactNotFoundError, FileNotFoundError) as exc:
            # Historical files may be deleted or moved while the pi output
            # survives. The correction is still valid; only its SE is unknown.
            return unavailable(f"Recorded sample artifact is unavailable ({exc}).")
        except (KeyError, AttributeError):
            # Older unit fixtures omit a materialized legacy sample object.
            if not isinstance(internal, ProbabilitySplitTranslator):
                return unavailable("Recorded sample cannot be inspected.")
        else:
            validate_unique(sampled, primary_key, role="recorded sample")
            if set(key_tuples(sampled, primary_key)) != set(
                key_tuples(audit_frame, primary_key)
            ):
                return unavailable("Audit keys disagree with the recorded sample.")
        if "strata" in by_source:
            try:
                recorded_strata = project.get_artifact(by_source["strata"])
            except (ArtifactNotFoundError, KeyError, FileNotFoundError) as exc:
                return unavailable(f"Recorded strata artifact is unavailable ({exc}).")
            if strata is not None and strata.artifact_id != recorded_strata.artifact_id:
                raise ArtifactError("Explicit strata conflict with TeAL sampling provenance.")
            strata = recorded_strata
        elif design == "stratified_srswor":
            return unavailable("Recorded stratified design has no strata source.")
    elif design is None:
        return unavailable(
            "No verified probability-sampling provenance; first-order inclusion "
            "probabilities alone do not identify the sampling design. "
            "Declare design for an externally randomized audit."
        )

    if design == "stratified_srswor" and strata is None:
        raise ValueError("design='stratified_srswor' requires full-frame strata=.")
    stratified_frame, stratified_field = None, None
    if design == "stratified_srswor":
        if tuple(str(k) for k in strata.primary_key) != primary_key:
            raise ArtifactError("Strata have different primary-key columns.")
        fields = [str(f) for f in strata.get_data_columns()]
        if len(fields) != 1:
            raise ArtifactError("Strata require one data field.")
        stratified_field = fields[0]
        try:
            stratified_frame = field_frame(strata, stratified_field)
        except (ArtifactNotFoundError, FileNotFoundError) as exc:
            if internal is not None:
                return unavailable(f"Recorded strata data are unavailable ({exc}).")
            raise
        validate_unique(stratified_frame, primary_key, role="strata")
        if set(key_tuples(stratified_frame, primary_key)) != population_keys:
            raise ArtifactError("Strata must cover every population key.")
        if stratified_frame[stratified_field].isna().any():
            raise ArtifactError(
                "Sampling strata contain missing or null values; "
                "every population unit must have a valid stratum."
            )
        from text_analysis_lab.core.probability_split import _stratum_scalar
        for value in stratified_frame[stratified_field]:
            try:
                _stratum_scalar(value)
            except (TypeError, ValueError) as exc:
                raise ArtifactError(
                    f"Sampling strata contain unsupported value {value!r}."
                ) from exc

    if design == "srswor":
        if not np.allclose(inclusion, n / N, rtol=1e-8, atol=1e-10):
            raise ArtifactError(f"SRSWOR pi contradicts n/N={n / N}.")
        v = srs_variance(residual, N=N, n=n)
        if v is None:
            return unavailable("An SRS audit of size one cannot estimate variance.")
        return v, "srswor", None

    if design == "stratified_srswor":
        joined = audit_frame.merge(
            stratified_frame.loc[:, [*primary_key, stratified_field]],
            on=list(primary_key), validate="one_to_one", sort=False,
        )
        size = stratified_frame.groupby(stratified_field, dropna=False).size()
        count = joined.groupby(stratified_field, dropna=False).size()
        for h, Nh in size.items():
            nh = int(count.get(h, 0))
            if nh == 0:
                raise ArtifactError(
                    f"Stratum {h!r} has no audit sample (zero inclusion probability)."
                )
            mask = (joined[stratified_field] == h).to_numpy()
            if not np.allclose(inclusion[mask], nh / Nh, rtol=1e-8, atol=1e-10):
                raise ArtifactError(
                    f"Stratum {h!r} pi contradicts n_h/N_h={nh / Nh}."
                )
        return stratified_variance(
            strata_frame=stratified_frame,
            audit_residuals=audit_frame.assign(__residual__=residual),
            primary_key=primary_key,
            stratum_field=stratified_field,
        )

    assert design == "bernoulli"
    if internal is not None:
        kind = internal.probability_kind
        if kind == "scalar":
            expected = np.full(n, internal.scalar_probability)
        else:
            try:
                if kind == "artifact":
                    artifact = project.get_artifact(by_source["probabilities"])
                elif kind == "strata":
                    artifact = project.get_artifact(by_source["strata"])
                else:
                    return unavailable("Unknown Bernoulli probability source.")
            except (ArtifactNotFoundError, KeyError, FileNotFoundError) as exc:
                return unavailable(f"Recorded Bernoulli source is unavailable ({exc}).")
            fields = [str(f) for f in artifact.get_data_columns()]
            if len(fields) != 1:
                return unavailable("Recorded probability source has ambiguous fields.")
            data = field_frame(artifact, fields[0])
            validate_unique(data, primary_key, role="probability inputs")
            if set(key_tuples(data, primary_key)) != population_keys:
                return unavailable("Probability input keys differ from the frame.")
            aligned = audit_frame.merge(
                data, on=list(primary_key), validate="one_to_one", sort=False
            )
            if kind == "artifact":
                expected = aligned[fields[0]].to_numpy(dtype=float)
            else:
                from text_analysis_lab.core.probability_split import _stratum_token
                mapping = {
                    _stratum_token(h): p for h, p in internal.stratum_probabilities
                }
                expected = np.asarray([
                    mapping[_stratum_token(h)] for h in aligned[fields[0]]
                ], dtype=float)
        if not np.allclose(inclusion, expected, rtol=1e-8, atol=1e-10):
            raise ArtifactError("Sampled pi contradicts recorded Bernoulli probabilities.")
    v = float(np.sum((1.0 - inclusion) * np.square(residual / inclusion)))
    note = None if internal is not None else (
        "SE assumes independent Bernoulli randomization and positivity "
        "for every frame unit; the external design is researcher asserted."
    )
    return v, "bernoulli", note
