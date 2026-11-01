"""IID-frame uncertainty for a probability-audited generalized-difference mean.

Assume N corpus units are IID and the surrogate is frozen independently of the
audit labels. Given the realized frame F, GD is design-unbiased for its mean:
Var(GD - mu) = Var_F(Y)/N + E_F[Var_design(GD | F)].

The corpus component estimates the *finite-frame* S_Y**2 / N unbiasedly from
observed audit pairs and their joint inclusion probabilities. This differs from
the independent labeled/unlabeled PPI variance decomposition. All three
supported designs have known pair probabilities; no general PPS design is
implicitly assumed.
"""
from __future__ import annotations

import numpy as np

from text_analysis_lab.core.errors import ArtifactError


def corpus_variance(
    *, gold: np.ndarray, inclusion: np.ndarray, population_size: int,
    design: str, stratum_population=None, stratum_audit=None,
) -> float:
    """Design-unbiased estimator of S_Y**2/N under a supported audit design.

    Bernoulli: pi_ij=pi_i*pi_j. SRS: pi_ij=n(n-1)/N(N-1).
    Stratified fixed-size: within h pi_ij=nh(nh-1)/Nh(Nh-1),
    between h,k pi_ij=(nh/Nh)(nk/Nk). Pair differences are nonnegative.
    """
    y = np.asarray(gold, dtype=float)
    pi = np.asarray(inclusion, dtype=float)
    N, n = int(population_size), len(y)
    if N < 2 or n < 2:
        raise ValueError(
            "Superpopulation corpus variance requires N >= 2 and at least "
            "two audited outcomes."
        )
    if not (len(pi) == n and np.all(np.isfinite(y)) and
            np.all(np.isfinite(pi)) and np.all((pi > 0) & (pi <= 1))):
        raise ArtifactError("Superpopulation audit outcomes/pi must be valid.")

    if design == "srswor":
        return float(np.var(y, ddof=1) / N)

    if design == "bernoulli":
        weights = 1.0 / pi
        total_weight = float(np.sum(weights))
        center = float(np.average(y, weights=weights))
        pair_sum = total_weight * float(
            np.sum(weights * np.square(y - center))
        )
        return float(pair_sum / (N * N * (N - 1)))

    if design != "stratified_srswor":
        raise ValueError(
            "Superpopulation inference requires a supported probability design."
        )
    if stratum_population is None or stratum_audit is None:
        raise ValueError(
            "Superpopulation stratified variance requires full population "
            "and audited stratum membership."
        )
    h_full = np.asarray(stratum_population, dtype=object)
    h_audit = np.asarray(stratum_audit, dtype=object)
    if len(h_full) != N or len(h_audit) != n:
        raise ArtifactError("Stratum memberships must align to population and audit.")
    # Existing design dispatch has validated nonnull scalar strata and keys.
    groups = []
    for h in dict.fromkeys(h_full.tolist()):
        values = y[h_audit == h]
        Nh, nh = int(np.count_nonzero(h_full == h)), len(values)
        if nh == 0 or (Nh > 1 and nh < 2):
            raise ValueError(
                f"Superpopulation variance cannot estimate within-stratum "
                f"outcome variation for stratum {h!r} with N_h={Nh}, n_h={nh}."
            )
        mean = float(np.mean(values))
        sample_var = float(np.var(values, ddof=1)) if nh > 1 else 0.0
        # E[s2_h] = S2_h, and E[sum_{i!=j in h} (yi-yj)^2] is
        # Nh*(Nh-1)*s2_h, where values denote an unordered pair sum.
        groups.append((
            Nh, mean, sample_var, (Nh - 1) * sample_var / Nh
        ))
    pair_sum = sum(Nh * (Nh - 1) * s2 for Nh, _, s2, _ in groups)
    for i, (Nh, mean_h, _, centered_h) in enumerate(groups):
        for Nk, mean_k, _, centered_k in groups[i + 1:]:
            pair_sum += Nh * Nk * (
                centered_h + centered_k + (mean_h - mean_k) ** 2
            )
    return float(pair_sum / (N * N * (N - 1)))
