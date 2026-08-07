"""
Distribution fitting utilities.

The `fitting` subpackage provides tools for estimating distribution
parameters from data and evaluating goodness‑of‑fit metrics.

Exports
-------
fit_distribution
    Fit a named distribution to data using SciPy MLE.
summarize_fit
    Produce a structured summary including log‑likelihood, AIC, BIC,
    and Kolmogorov–Smirnov statistics.

Notes
-----
- All fitting results use the `FitResult` TypedDict defined in `types.py`.
- The API is designed to integrate with the `distributions` subpackage.
"""

from .fitting import fit_distribution, summarize_fit

__all__ = [
    "fit_distribution",
    "summarize_fit",
]
