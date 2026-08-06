"""
Distribution fitting utilities.

Υποστηρίζει:
- MLE fitting μέσω SciPy .fit()
- log-likelihood
- AIC / BIC
- KS goodness-of-fit
- unified API για fit(), summarize(), plot()

Κάθε συνάρτηση επιστρέφει FitResult:
{
    "params": {...},
    "loglik": float,
    "aic": float,
    "bic": float,
    "ks_stat": float,
    "ks_pvalue": float,
    "method": str,
    "extra": dict
}
"""

from collections.abc import Sequence
from typing import Any, cast

import numpy as np
from numpy import ndarray
from scipy.stats import (
    kstest,
    rv_continuous,
    rv_discrete,
)
from scipy.stats._distn_infrastructure import rv_frozen

from .types import FitResult

NumericArray = ndarray | Sequence[float]
DistributionType = rv_continuous | rv_discrete | rv_frozen


# ---------------------------------------------------------
# Log-likelihood
# ---------------------------------------------------------


def _loglik(dist: rv_frozen, data: ndarray) -> float:
    pdf_vals = cast(Any, dist).pdf(data)
    pdf_vals = np.clip(pdf_vals, 1e-300, None)
    return float(np.sum(np.log(pdf_vals)))


# ---------------------------------------------------------
# AIC / BIC
# ---------------------------------------------------------


def _aic(loglik: float, k: int) -> float:
    return float(2 * k - 2 * loglik)


def _bic(loglik: float, k: int, n: int) -> float:
    return float(np.log(n) * k - 2 * loglik)


# ---------------------------------------------------------
# Main fitting function
# ---------------------------------------------------------


def fit_distribution(
    dist: DistributionType, data: NumericArray, method: str = "MLE"
) -> FitResult:
    """
    Fit a SciPy distribution using MLE (.fit()).

    Parameters
    ----------
    dist : SciPy distribution (norm, gamma, beta, etc.)
    data : array-like
    method : str
        Currently only 'MLE' is supported.

    Returns
    -------
    FitResult
    """

    arr = np.asarray(data, dtype=float)

    # Fit parameters via MLE
    params = cast(Any, dist).fit(arr)

    # Create frozen distribution with fitted params
    fitted = cast(Any, dist)(*params)

    # Compute log-likelihood
    loglik = _loglik(fitted, arr)

    # Number of parameters
    k = len(params)

    # AIC / BIC
    aic = _aic(loglik, k)
    bic = _bic(loglik, k, len(arr))

    # KS goodness-of-fit
    ks_stat, ks_pvalue = kstest(arr, cast(Any, fitted).cdf)

    # Distribution name
    dist_name = dist.__class__.__name__

    # ---------------------------------------------------------
    # Parameter names (shape parameters + loc + scale)
    # ---------------------------------------------------------

    # SciPy .fit() always returns:
    # (shape1, shape2, ..., loc, scale)
    num_params = len(params)

    if num_params < 2:
        raise RuntimeError("Invalid parameter count returned by SciPy .fit()")

    # SciPy .fit() returns (shape..., loc, scale)
    num_params = len(params)

    if num_params < 2:
        raise RuntimeError("Invalid parameter count returned by SciPy .fit()")

    # Shape parameters = all except last two
    num_shapes = num_params - 2
    shape_names = [f"shape{i + 1}" for i in range(num_shapes)]

    # Final parameter names
    names = shape_names + ["loc", "scale"]

    # Build parameter dictionary
    param_dict = {name: float(val) for name, val in zip(names, params)}

    return {
        "params": param_dict,
        "loglik": loglik,
        "aic": aic,
        "bic": bic,
        "ks_stat": float(ks_stat),
        "ks_pvalue": float(ks_pvalue),
        "method": method,
        "extra": {"distribution": dist_name},
    }


# ---------------------------------------------------------
# Summary utility
# ---------------------------------------------------------


def summarize_fit(result: FitResult) -> dict[str, float]:
    """
    Return a compact summary of the fitted distribution.
    """
    return {
        "loglik": result["loglik"],
        "aic": result["aic"],
        "bic": result["bic"],
        "ks_stat": result["ks_stat"],
        "ks_pvalue": result["ks_pvalue"],
    }
