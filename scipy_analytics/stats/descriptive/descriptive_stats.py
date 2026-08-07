"""
Descriptive statistics utilities.

This module provides a unified API for computing descriptive statistics
(mean, median, mode, variance, standard deviation, skewness, kurtosis,
trimmed mean, winsorized mean, geometric mean, harmonic mean) using NumPy
and SciPy.

Each low‑level function returns a `LowLevelResult`:

    {
        "stat": float,
        "method": str,
        "extra": dict
    }

The high‑level function `descriptive_stats` returns a `DescriptiveResult`
containing a full summary of descriptive metrics.

Features
--------
- Unified return type for all descriptive metrics
- NumPy‑friendly API
- Robust statistics (trimmed mean, winsorized mean)
- Optional kurtosis computation
- Percentile summary (0, 25, 50, 75, 100)
- Safe handling of empty input

Notes
-----
- All functions accept Python sequences or NumPy arrays.
- All results are cast to Python floats for consistency.
- Variance and standard deviation use ddof=1 (sample statistics).
"""

from collections.abc import Sequence

import numpy as np
from numpy import ndarray
from scipy.stats import (
    gmean,
    hmean,
    kurtosis,
    mode,
    mstats,
    skew,
    trim_mean,
)

from .types import DescriptiveResult, LowLevelResult

NumericArray = ndarray | Sequence[float]


def mean(data: NumericArray) -> LowLevelResult:
    """
    Compute the arithmetic mean.

    Parameters
    ----------
    data : NumericArray
        Sequence of numeric values.

    Returns
    -------
    LowLevelResult
        stat : float
            The mean value.
        method : "mean"
        extra : {}

    Notes
    -----
    - Equivalent to NumPy's `np.mean`.
    - Sensitive to outliers.
    """
    value = float(np.mean(data))
    return {"stat": value, "method": "mean", "extra": {}}


def median(data: NumericArray) -> LowLevelResult:
    """
    Compute the median.

    The median is robust to outliers and represents the central point
    of the distribution.

    Parameters
    ----------
    data : NumericArray

    Returns
    -------
    LowLevelResult
    """
    value = float(np.median(data))
    return {"stat": value, "method": "median", "extra": {}}


def mode_value(data: NumericArray) -> LowLevelResult:
    """
    Compute the mode (most frequent value).

    Returns
    -------
    LowLevelResult
        extra["count"] contains the number of occurrences.

    Notes
    -----
    - Uses SciPy's `mode` with `keepdims=True`.
    """
    m = mode(data, keepdims=True)
    return {
        "stat": float(m.mode[0]),
        "method": "mode",
        "extra": {"count": int(m.count[0])},
    }


def variance(data: NumericArray, ddof: int = 1) -> LowLevelResult:
    """
    Compute sample variance.

    Parameters
    ----------
    data : NumericArray
    ddof : int
        Degrees of freedom (default 1 for sample variance).

    Returns
    -------
    LowLevelResult

    Notes
    -----
    - Equivalent to `np.var(data, ddof=1)`.
    """
    value = float(np.var(data))
    return {"stat": value, "method": "variance", "extra": {"ddof": ddof}}


def std(data: NumericArray, ddof: int = 1) -> LowLevelResult:
    """
    Compute sample standard deviation.

    Returns
    -------
    LowLevelResult

    Notes
    -----
    - Equivalent to `np.std(data, ddof=1)`.
    """
    value = float(np.std(data))
    return {"stat": value, "method": "std", "extra": {"ddof": ddof}}


def skewness(data: NumericArray) -> LowLevelResult:
    """
    Compute skewness (asymmetry of the distribution).

    Returns
    -------
    LowLevelResult

    Notes
    -----
    - Uses SciPy's `skew`.
    - Positive skew → right tail heavier.
    - Negative skew → left tail heavier.
    """
    value = float(skew(data))
    return {"stat": value, "method": "skewness", "extra": {}}


def kurtosis_value(data: NumericArray) -> LowLevelResult:
    """
    Compute kurtosis (tail heaviness).

    Returns
    -------
    LowLevelResult

    Notes
    -----
    - Uses SciPy's `kurtosis` (Fisher definition).
    - Normal distribution → kurtosis = 0.
    """
    value = float(kurtosis(data))
    return {"stat": value, "method": "kurtosis", "extra": {}}


def trimmed_mean_value(data: NumericArray, proportion: float = 0.1) -> LowLevelResult:
    """
    Compute trimmed mean.

    Removes a proportion of low and high values before computing the mean.

    Parameters
    ----------
    proportion : float
        Fraction trimmed from each tail.

    Returns
    -------
    LowLevelResult

    Notes
    -----
    - Robust to outliers.
    """
    value = float(trim_mean(data, proportion))
    return {
        "stat": value,
        "method": "trimmed_mean",
        "extra": {"proportion": proportion},
    }


def winsorized_mean(
    data: NumericArray, limits: tuple[float, float] = (0.1, 0.1)
) -> LowLevelResult:
    """
    Compute winsorized mean.

    Instead of removing outliers, replaces them with boundary values.

    Parameters
    ----------
    limits : tuple[float, float]
        Winsorization proportions for low and high tails.

    Returns
    -------
    LowLevelResult

    Notes
    -----
    - Uses SciPy's `mstats.winsorize`.
    """
    w = mstats.winsorize(data, limits=limits)
    value = float(np.mean(w))
    return {
        "stat": value,
        "method": "winsorized_mean",
        "extra": {"limits": limits},
    }


def geometric_mean(data: NumericArray) -> LowLevelResult:
    """
    Compute geometric mean.

    Returns
    -------
    LowLevelResult

    Notes
    -----
    - Suitable for multiplicative processes.
    - All values must be positive.
    """
    value = float(gmean(data))
    return {"stat": value, "method": "geometric_mean", "extra": {}}


def harmonic_mean(data: NumericArray) -> LowLevelResult:
    """
    Compute harmonic mean.

    Returns
    -------
    LowLevelResult

    Notes
    -----
    - Suitable for rates (e.g., speed, ratios).
    - All values must be positive.
    """
    value = float(hmean(data))
    return {"stat": value, "method": "harmonic_mean", "extra": {}}


def descriptive_stats(data: NumericArray, kurt: bool = False) -> DescriptiveResult:
    """
    Compute a full descriptive statistics summary.

    Parameters
    ----------
    data : NumericArray
        Sequence of numeric values.
    kurt : bool
        Whether to include kurtosis in the result.

    Returns
    -------
    DescriptiveResult
        Dictionary containing:
        - mean
        - median
        - mode
        - variance
        - std
        - skewness
        - min, max
        - count
        - percentiles (0, 25, 50, 75, 100)
        - kurtosis (optional)

    Raises
    ------
    ValueError
        If input is empty.

    Notes
    -----
    - All values are cast to Python floats.
    - Percentiles use NumPy's `np.percentile`.
    """
    arr = np.asarray(data, dtype=float)
    if arr.size == 0:
        raise ValueError("Empty input")

    return {
        "mean": mean(arr)["stat"],
        "median": median(arr)["stat"],
        "mode": mode_value(arr)["stat"],
        "variance": variance(arr)["stat"],
        "std": std(arr)["stat"],
        "skewness": skewness(arr)["stat"],
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "count": int(arr.size),
        "percentiles": {
            0: float(np.percentile(arr, 0)),
            25: float(np.percentile(arr, 25)),
            50: float(np.percentile(arr, 50)),
            75: float(np.percentile(arr, 75)),
            100: float(np.percentile(arr, 100)),
        },
        "kurtosis": kurtosis_value(arr)["stat"] if kurt else None,
    }
