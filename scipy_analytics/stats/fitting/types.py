"""
Typed result structures for distribution fitting.

This module defines the `FitResult` TypedDict, which provides a unified
structure for returning distribution fitting results. It is used by
`fit_distribution` and `summarize_fit` in the fitting subpackage.

The goal is to provide:
- consistent return format
- compatibility with static type checkers (mypy, Pylance)
- clear programmatic access to all fitting metrics
"""

from typing import Any, TypedDict


class FitResult(TypedDict):
    """
    Structured result for distribution fitting.

    Keys
    ----
    params : dict[str, float]
        Estimated distribution parameters (shape, loc, scale).
    loglik : float
        Log‑likelihood of the fitted model.
    aic : float
        Akaike Information Criterion.
    bic : float
        Bayesian Information Criterion.
    ks_stat : float
        Kolmogorov–Smirnov test statistic.
    ks_pvalue : float
        Kolmogorov–Smirnov p‑value.
    method : str
        Name of the fitting method (e.g., "MLE", "KS", "grid_search").
    extra : dict[str, Any]
        Additional metadata (e.g., convergence info, warnings).
    """

    params: dict[str, float]
    loglik: float
    aic: float
    bic: float
    ks_stat: float
    ks_pvalue: float
    method: str
    extra: dict[str, Any]
