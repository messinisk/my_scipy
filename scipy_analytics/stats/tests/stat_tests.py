"""
Statistical hypothesis testing utilities.

Πλήρης υλοποίηση των πιο συνηθισμένων SciPy tests με unified return format.

Κάθε συνάρτηση επιστρέφει TestResult:
{
    "statistic": float,
    "pvalue": float,
    "method": str,
    "extra": dict
}

Στόχος:
- Ενοποιημένο API για hypothesis testing
- Plug-and-play χρήση σε pipelines και reports
- Συμβατότητα με scipy.stats
"""

from collections.abc import Sequence

import numpy as np
from numpy import ndarray
from scipy.stats import (
    anderson,
    fisher_exact,
    friedmanchisquare,
    jarque_bera,
    ks_2samp,
    kstest,
    levene,
    mannwhitneyu,
    normaltest,
    shapiro,
    ttest_ind,
    ttest_rel,
    wilcoxon,
)

from .types import TestResult

NumericArray = ndarray | Sequence[float]


# -----------------------------
# Parametric tests
# -----------------------------


def ttest_independent(a: NumericArray, b: NumericArray) -> TestResult:
    """Independent samples t-test."""
    stat, p = ttest_ind(a, b)
    return {
        "statistic": float(stat),
        "pvalue": float(p),
        "method": "ttest_ind",
        "extra": {},
    }


def ttest_paired(a: NumericArray, b: NumericArray) -> TestResult:
    """Paired samples t-test."""
    stat, p = ttest_rel(a, b)
    return {
        "statistic": float(stat),
        "pvalue": float(p),
        "method": "ttest_rel",
        "extra": {},
    }


# -----------------------------
# Non-parametric tests
# -----------------------------


def mann_whitney(a: NumericArray, b: NumericArray) -> TestResult:
    """Mann–Whitney U test."""
    stat, p = mannwhitneyu(a, b)
    return {
        "statistic": float(stat),
        "pvalue": float(p),
        "method": "mann_whitney",
        "extra": {},
    }


def wilcoxon_test(a: NumericArray, b: NumericArray) -> TestResult:
    """Wilcoxon signed-rank test."""
    stat, p = wilcoxon(a, b)
    return {
        "statistic": float(stat),
        "pvalue": float(p),
        "method": "wilcoxon",
        "extra": {},
    }


# -----------------------------
# Normality tests
# -----------------------------


def shapiro_test(data: NumericArray) -> TestResult:
    """Shapiro–Wilk normality test."""
    stat, p = shapiro(data)
    return {
        "statistic": float(stat),
        "pvalue": float(p),
        "method": "shapiro",
        "extra": {},
    }


def jarque_bera_test(data: NumericArray) -> TestResult:
    """Jarque–Bera normality test."""
    stat, p = jarque_bera(data)
    return {
        "statistic": float(stat),
        "pvalue": float(p),
        "method": "jarque_bera",
        "extra": {},
    }


def dagostino_test(data: NumericArray) -> TestResult:
    """D’Agostino’s K² normality test."""
    stat, p = normaltest(data)
    return {
        "statistic": float(stat),
        "pvalue": float(p),
        "method": "dagostino_k2",
        "extra": {},
    }


def anderson_test(data: NumericArray) -> TestResult:
    """Anderson–Darling test."""
    result = anderson(data)
    return {
        "statistic": float(result.statistic),
        "pvalue": float("nan"),
        "method": "anderson",
        "extra": {
            "critical_values": result.critical_values,
            "significance": result.significance_level,
        },
    }


# -----------------------------
# Variance tests
# -----------------------------


def levene_test(a: NumericArray, b: NumericArray) -> TestResult:
    """Levene variance test."""
    stat, p = levene(a, b)
    return {
        "statistic": float(stat),
        "pvalue": float(p),
        "method": "levene",
        "extra": {},
    }


# -----------------------------
# Goodness-of-fit tests
# -----------------------------


def ks_test(a: NumericArray, b: NumericArray) -> TestResult:
    """Kolmogorov–Smirnov test (two-sample)."""
    stat, p = ks_2samp(a, b)
    return {
        "statistic": float(stat),
        "pvalue": float(p),
        "method": "ks_2samp",
        "extra": {},
    }


def kstest_distribution(data: NumericArray, dist: str = "norm") -> TestResult:
    """One-sample KS test against a distribution."""
    stat, p = kstest(data, dist)
    return {
        "statistic": float(stat),
        "pvalue": float(p),
        "method": "kstest",
        "extra": {"dist": dist},
    }


# -----------------------------
# Contingency tests
# -----------------------------

# def chi_square(table: Sequence[Sequence[int]]) -> TestResult:
#     """
#     Chi-square test of independence.

#     Μετατροπή σε NumPy array για συμβατότητα με typing stubs.
#     """
#     arr = np.asarray(table, dtype=float)
#     stat, p, dof, expected = chi2_contingency(arr)
#     return {
#         "statistic": float(stat),
#         "pvalue": float(p),
#         "method": "chi2_contingency",
#         "extra": {"dof": dof, "expected": expected},
#     }


def fisher_test(table: Sequence[Sequence[int]]) -> TestResult:
    """
    Fisher exact test.

    SciPy δέχεται πίνακες 2x2 ως Python lists,
    αλλά τα typing stubs απαιτούν NumPy array.
    Γι' αυτό μετατρέπουμε το input σε ndarray.
    """
    arr = np.asarray(table, dtype=float)
    stat, p = fisher_exact(arr)
    return {
        "statistic": float(stat),
        "pvalue": float(p),
        "method": "fisher_exact",
        "extra": {},
    }


# -----------------------------
# Friedman test
# -----------------------------


def friedman_test(*groups: NumericArray) -> TestResult:
    """Friedman test for repeated measures."""
    stat, p = friedmanchisquare(*groups)
    return {
        "statistic": float(stat),
        "pvalue": float(p),
        "method": "friedman",
        "extra": {},
    }
