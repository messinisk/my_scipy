"""
Contingency table analysis utilities.

Υποστηρίζει:
- χ² test of independence
- odds ratio
- Cramér’s V
- φ-coefficient
- table utilities

Κάθε συνάρτηση επιστρέφει ContingencyResult:
{
    "statistic": float,
    "pvalue": float,
    "method": str,
    "extra": dict
}
"""

from collections.abc import Sequence

import numpy as np
from scipy.stats import chi2_contingency, fisher_exact

from .types import ContingencyResult

ContingencyTable = Sequence[Sequence[int]] | np.ndarray


def _to_array(table: ContingencyTable) -> np.ndarray:
    return np.asarray(table, dtype=float)


def chi_square(table: ContingencyTable) -> ContingencyResult:
    arr = _to_array(table)
    stat, p, dof, expected = chi2_contingency(arr)
    return {
        "statistic": float(stat),
        "pvalue": float(p),
        "method": "chi_square",
        "extra": {"dof": dof, "expected": expected},
    }


def odds_ratio(table: ContingencyTable) -> ContingencyResult:
    arr = _to_array(table)
    if arr.shape != (2, 2):
        raise ValueError("Odds ratio requires a 2x2 table.")
    or_value, p = fisher_exact(arr)
    return {
        "statistic": float(or_value),
        "pvalue": float(p),
        "method": "odds_ratio",
        "extra": {},
    }


def phi_coefficient(table: ContingencyTable) -> ContingencyResult:
    arr = _to_array(table)
    if arr.shape != (2, 2):
        raise ValueError("Phi coefficient requires a 2x2 table.")
    chi = chi_square(arr)
    n = arr.sum()
    phi = np.sqrt(chi["statistic"] / n)
    return {
        "statistic": float(phi),
        "pvalue": chi["pvalue"],
        "method": "phi_coefficient",
        "extra": {"n": n},
    }


def cramers_v(table: ContingencyTable) -> ContingencyResult:
    arr = _to_array(table)
    chi = chi_square(arr)
    n = arr.sum()
    r, c = arr.shape
    denom = n * (min(r - 1, c - 1))
    V = np.sqrt(chi["statistic"] / denom)
    return {
        "statistic": float(V),
        "pvalue": chi["pvalue"],
        "method": "cramers_v",
        "extra": {"n": n, "rows": r, "cols": c},
    }
