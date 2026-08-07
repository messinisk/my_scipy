"""
Typed result structures for descriptive statistics.

This module defines two TypedDict classes used throughout the
`descriptive_stats` subpackage:

- LowLevelResult: unified structure returned by individual descriptive
  statistic functions (mean, median, variance, etc.).
- DescriptiveResult: high‑level summary returned by `descriptive_stats`,
  containing all major descriptive metrics.

These types ensure:
- consistent return format
- compatibility with static type checkers (mypy, Pylance)
- clarity when consuming results programmatically
"""

from typing import Any, TypedDict


class DescriptiveResult(TypedDict):
    """
    High‑level descriptive statistics summary.

    Returned by `descriptive_stats(data)`.

    Keys
    ----
    mean : float
        Arithmetic mean.
    median : float
        Median value.
    mode : float
        Most frequent value.
    variance : float
        Sample variance (ddof=1).
    std : float
        Sample standard deviation (ddof=1).
    skewness : float
        Distribution asymmetry.
    min : float
        Minimum value.
    max : float
        Maximum value.
    count : int
        Number of observations.
    percentiles : dict[int, float]
        Percentile values at 0, 25, 50, 75, 100.
    kurtosis : float | None
        Kurtosis (optional, only if `kurt=True`).
    """

    mean: float
    median: float
    mode: float
    variance: float
    std: float
    skewness: float
    min: float
    max: float
    count: int
    percentiles: dict[int, float]
    kurtosis: float | None


class LowLevelResult(TypedDict):
    """
    Unified structure returned by low‑level descriptive functions.

    Example:
        {
            "stat": 3.14,
            "method": "mean",
            "extra": {}
        }

    Keys
    ----
    stat : float
        The computed statistic.
    method : str
        Name of the method (e.g., "mean", "median").
    extra : dict[str, Any]
        Additional metadata (e.g., ddof, trimming proportion).
    """

    stat: float
    method: str
    extra: dict[str, Any]
