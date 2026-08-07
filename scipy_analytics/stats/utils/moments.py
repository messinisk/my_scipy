"""
Sample moments computation utilities.

This module provides the `sample_moments` function, which computes the
first four central moments of a sample:

- mean
- variance
- skewness
- kurtosis

These quantities are fundamental in descriptive statistics and are used
in the method of moments for parameter estimation in probability
distributions (including Pearson distributions).

Definitions
-----------
Given a sample X = {x₁, x₂, ..., xₙ}:

Mean:
    μ = (1/n) Σ xᵢ

Variance:
    σ² = (1/n) Σ (xᵢ - μ)²

Skewness:
    γ₁ = E[(X - μ)³] / σ³

Kurtosis:
    γ₂ = E[(X - μ)⁴] / σ⁴

Notes
-----
- This implementation uses population moments (normalizing by n).
- For statistical inference, SciPy's `skew` and `kurtosis` may be preferred.
- Variance must be non-zero for skewness and kurtosis to be defined.
"""

from __future__ import annotations

import numpy as np


def sample_moments(data):
    """
    Compute the first four sample moments: mean, variance, skewness, kurtosis.

    Parameters
    ----------
    data : array_like
        Input numeric sample.

    Returns
    -------
    tuple
        (mean, variance, skewness, kurtosis)

    Raises
    ------
    ValueError
        If variance is zero (skewness and kurtosis undefined).

    Examples
    --------
    >>> import numpy as np
    >>> from scipy_analytics.stats.utils.moments import sample_moments
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> sample_moments(x)
    (3.0, 2.0, 0.0, 1.7)

    Notes
    -----
    - Uses population variance (normalization by n).
    - Skewness and kurtosis follow the standard central moment definitions.
    """
    data = np.asarray(data, dtype=float)

    mean = np.mean(data)
    var = np.var(data)

    if var == 0:
        raise ValueError("Variance is zero; skewness and kurtosis undefined.")

    skew = ((data - mean) ** 3).mean() / var**1.5
    kurt = ((data - mean) ** 4).mean() / var**2

    return mean, var, skew, kurt
