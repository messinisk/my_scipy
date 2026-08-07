"""
Unified API for probability distributions.

This module provides a simple, name‑based interface for evaluating
probability distributions. Instead of instantiating SciPy distribution
objects manually, users can call:

    pdf("normal", x, loc=0, scale=1)
    cdf("gamma", x, a=2, scale=1)
    ppf("beta", q, a=2, b=5)
    rvs("t", size=100, df=10)
    stats("lognorm", s=0.5)

The API delegates distribution lookup to `get_dist(name)` from
`distributions.core`, which returns a SciPy distribution class or a
custom wrapper.

Features
--------
- Unified functional API for all distributions
- Name‑based lookup (string → distribution)
- Consistent parameter passing via **params
- NumPy‑friendly return types
- Supports all SciPy continuous distributions registered in core.py

Notes
-----
- All functions return NumPy arrays or Python floats.
- Parameters must match the underlying SciPy distribution signature.
- `get_dist(name)` must be configured in `core.py` for each supported
  distribution.
"""

from typing import Any

import numpy as np

from .core import get_dist


def pdf(name: str, x: Any, **params: Any) -> np.ndarray:
    """
    Evaluate the probability density function (PDF).

    Parameters
    ----------
    name : str
        Name of the distribution (e.g., "normal", "gamma", "beta").
    x : array-like
        Points at which to evaluate the PDF.
    **params : Any
        Distribution parameters (shape, loc, scale).

    Returns
    -------
    np.ndarray
        PDF values.

    Examples
    --------
    >>> pdf("normal", [0, 1], loc=0, scale=1)
    array([0.3989, 0.2419])
    """
    dist = get_dist(name)(**params)
    return dist.pdf(x)  # type: ignore


def cdf(name: str, x: Any, **params: Any) -> np.ndarray:
    """
    Evaluate the cumulative distribution function (CDF).

    Parameters
    ----------
    name : str
        Distribution name.
    x : array-like
        Points at which to evaluate the CDF.
    **params : Any
        Distribution parameters.

    Returns
    -------
    np.ndarray
        CDF values.

    Examples
    --------
    >>> cdf("gamma", [0.5, 1.0], a=2)
    array([0.0902, 0.2642])
    """
    dist = get_dist(name)(**params)
    return dist.cdf(x)  # type: ignore


def ppf(name: str, q: Any, **params: Any) -> np.ndarray:
    """
    Evaluate the percent point function (inverse CDF).

    Parameters
    ----------
    name : str
        Distribution name.
    q : array-like
        Quantiles in [0, 1].
    **params : Any
        Distribution parameters.

    Returns
    -------
    np.ndarray
        Quantile values.

    Examples
    --------
    >>> ppf("beta", [0.1, 0.9], a=2, b=5)
    array([0.150, 0.622])
    """
    dist = get_dist(name)(**params)
    return dist.ppf(q)  # type: ignore


def rvs(name: str, size: int = 1, **params: Any) -> np.ndarray:
    """
    Draw random samples from a distribution.

    Parameters
    ----------
    name : str
        Distribution name.
    size : int
        Number of samples.
    **params : Any
        Distribution parameters.

    Returns
    -------
    np.ndarray
        Random samples.

    Examples
    --------
    >>> rvs("normal", size=3, loc=0, scale=1)
    array([ 0.12, -0.44, 1.03])
    """
    dist = get_dist(name)(**params)
    return dist.rvs(size=size)  # type: ignore


def stats(name: str, **params: Any) -> dict[str, float]:
    """
    Compute distribution summary statistics.

    Parameters
    ----------
    name : str
        Distribution name.
    **params : Any
        Distribution parameters.

    Returns
    -------
    dict[str, float]
        Dictionary containing:
        - mean
        - var
        - skew
        - kurtosis

    Examples
    --------
    >>> stats("normal", loc=0, scale=1)
    {'mean': 0.0, 'var': 1.0, 'skew': 0.0, 'kurtosis': 0.0}

    Notes
    -----
    - Uses SciPy's `dist.stats(moments="mvsk")`.
    - All values are cast to Python floats.
    """
    dist = get_dist(name)(**params)
    mean, var, skew, kurt = dist.stats(moments="mvsk")
    return {
        "mean": float(mean),
        "var": float(var),
        "skew": float(skew),
        "kurtosis": float(kurt),
    }
