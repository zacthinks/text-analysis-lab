"""Explicit opt-in adoption of already fitted predictors at Project.translate.

This boundary adapter does not participate in sklearn fitting or change TeAL's
existing translator execution/freeze contracts.
"""

from __future__ import annotations

from typing import Any

from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from text_analysis_lab.core.errors import OperatorNotFittedError
from text_analysis_lab.translators.fitted_predictor import FittedPredictor


def adapt_fitted_predictor(model: Any) -> FittedPredictor:
    """Wrap a fitted sklearn(-like) predict object for durable TeAL execution."""
    predict = getattr(model, "predict", None)
    if not callable(predict):
        raise TypeError(
            "Project.translate expects a TeAL BaseTranslator or an already fitted "
            "predictor exposing predict(...)."
        )
    try:
        check_is_fitted(model)
    except (NotFittedError, TypeError, AttributeError) as exc:
        raise OperatorNotFittedError(
            "Project.translate received a predictor that cannot be verified as "
            "fitted. Fit it with sklearn before passing it to TeAL, or use an "
            "explicit FittedPredictor for a custom predictor."
        ) from exc
    return FittedPredictor(model, save_model=True)
