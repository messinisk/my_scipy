"""
Typed result structures for statistical hypothesis testing.

This module defines the `TestResult` TypedDict, which provides a unified
structure for returning results from hypothesis tests such as:

- t-test (one-sample, two-sample)
- Kolmogorov–Smirnov test
- χ² goodness-of-fit test
- Mann–Whitney U test
- Wilcoxon signed-rank test
- Shapiro–Wilk normality test
- Anderson–Darling test
- any custom test implemented in the `tests` subpackage

The goal is to ensure:
- consistent return format across all tests
- compatibility with static type checkers (mypy, Pylance)
- clear programmatic access to test statistics, p-values, and metadata
- extensibility via the `extra` field
"""

from typing import Any, TypedDict


class TestResult(TypedDict):
    """
    Structured result for hypothesis testing.

    Keys
    ----
    statistic : float
        The computed test statistic (e.g., t-value, KS statistic, χ² value).
    pvalue : float
        The p-value associated with the test statistic.
        Indicates the probability of observing the result under the null hypothesis.
    method : str
        Name of the statistical test (e.g., "t-test", "ks-test", "chi2-test").
    extra : dict[str, Any]
        Additional metadata, such as:
        - "df": degrees of freedom (for t-test, χ² test)
        - "alternative": alternative hypothesis ("two-sided", "greater", "less")
        - "n": sample size
        - "distribution": theoretical distribution used
        - any other method-specific information

    Examples
    --------
    >>> result = {
    ...     "statistic": 2.13,
    ...     "pvalue": 0.034,
    ...     "method": "t-test",
    ...     "extra": {"df": 28, "alternative": "two-sided"}
    ... }
    >>> result["pvalue"]
    0.034

    Notes
    -----
    - All values are cast to Python floats for consistency.
    - The `extra` field allows downstream code to adapt behavior depending
      on the test type and configuration.
    - This structure is used by all functions in `scipy_analytics.stats.tests`.
    """

    statistic: float
    pvalue: float
    method: str
    extra: dict[str, Any]
