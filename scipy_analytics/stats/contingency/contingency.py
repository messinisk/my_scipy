"""
Contingency table analysis utilities.

This module provides statistical measures and tests for categorical data
organized in contingency tables. It includes:

- Chi-square test of independence
- Odds ratio (2×2 tables)
- Phi coefficient (2×2 tables)
- Cramér’s V
- Tschuprow’s T
- Table conversion utilities

Each function returns a `ContingencyResult` TypedDict:

    {
        "statistic": float,
        "pvalue": float,
        "method": str,
        "extra": dict
    }

All functions accept either:
- nested Python sequences (lists of lists), or
- NumPy arrays

and internally convert them to `np.ndarray`.
"""

from collections.abc import Sequence
from typing import Any, TypedDict

import numpy as np
from scipy.stats import chi2_contingency, fisher_exact

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class ContingencyResult(TypedDict):
    """Typed result for contingency table statistics."""

    statistic: float
    pvalue: float
    method: str
    extra: dict[str, Any]


ContingencyTable = Sequence[Sequence[int]] | np.ndarray


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _to_array(table: ContingencyTable) -> np.ndarray:
    """
    Convert input table to a NumPy array of floats.

    Parameters
    ----------
    table : ContingencyTable
        A nested sequence or NumPy array representing a contingency table.

    Returns
    -------
    np.ndarray
        A 2D array of floats.
    """
    return np.asarray(table, dtype=float)


# ---------------------------------------------------------------------------
# Chi-square test
# ---------------------------------------------------------------------------


def chi_square(table: ContingencyTable) -> ContingencyResult:
    """
    Perform the chi-square test of independence on a contingency table.

    Uses `scipy.stats.chi2_contingency` with Yates' correction enabled by
    default for 2×2 tables.

    Parameters
    ----------
    table : ContingencyTable
        A 2D contingency table of non-negative counts.

    Returns
    -------
    ContingencyResult
        statistic : float
            The chi-square statistic.
        pvalue : float
            The p-value of the test.
        method : str
            Always "chi_square".
        extra : dict
            Contains:
                - "dof": degrees of freedom
                - "expected": expected frequencies under independence

    Notes
    -----
    - Expected frequencies may be fractional.
    - For small samples, Fisher's exact test may be more appropriate.
    """
    arr = _to_array(table)
    stat, p, dof, expected = chi2_contingency(arr)
    return {
        "statistic": float(stat),
        "pvalue": float(p),
        "method": "chi_square",
        "extra": {"dof": dof, "expected": expected},
    }


# ---------------------------------------------------------------------------
# Odds ratio (2×2)
# ---------------------------------------------------------------------------


def odds_ratio(table: ContingencyTable) -> ContingencyResult:
    """
    Compute the odds ratio for a 2×2 contingency table using Fisher's exact test.

    Parameters
    ----------
    table : ContingencyTable
        Must be a 2×2 table.

    Returns
    -------
    ContingencyResult
        statistic : float
            The odds ratio.
        pvalue : float
            The p-value from Fisher's exact test.
        method : str
            Always "odds_ratio".
        extra : dict
            Empty.

    Raises
    ------
    ValueError
        If the table is not 2×2.

    Notes
    -----
    - Odds ratio is undefined for larger tables.
    - Fisher's exact test is exact and appropriate for small samples.
    """
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


# ---------------------------------------------------------------------------
# Phi coefficient (2×2)
# ---------------------------------------------------------------------------


def phi_coefficient(table: ContingencyTable) -> ContingencyResult:
    """
    Compute the phi coefficient for a 2×2 contingency table.

    Phi is defined as:

        φ = sqrt( χ² / n )

    where χ² is the chi-square statistic and n is the total count.

    Parameters
    ----------
    table : ContingencyTable
        Must be a 2×2 table.

    Returns
    -------
    ContingencyResult
        statistic : float
            Phi coefficient.
        pvalue : float
            P-value from the chi-square test.
        method : str
            Always "phi_coefficient".
        extra : dict
            Contains:
                - "n": total sample size

    Raises
    ------
    ValueError
        If the table is not 2×2.
    """
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


# ---------------------------------------------------------------------------
# Cramér’s V
# ---------------------------------------------------------------------------


def cramers_v(table: ContingencyTable) -> ContingencyResult:
    """
    Compute Cramér’s V statistic for association between categorical variables.

    Defined as:

        V = sqrt( χ² / ( n * min(r - 1, c - 1) ) )

    Parameters
    ----------
    table : ContingencyTable
        A 2D contingency table.

    Returns
    -------
    ContingencyResult
        statistic : float
            Cramér’s V in [0, 1].
        pvalue : float
            P-value from the chi-square test.
        method : str
            Always "cramers_v".
        extra : dict
            Contains:
                - "n": total sample size
                - "rows": r
                - "cols": c

    Notes
    -----
    - Symmetric measure.
    - Suitable for rectangular tables.
    """
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


# ---------------------------------------------------------------------------
# Tschuprow’s T
# ---------------------------------------------------------------------------


def tschuprow_t(table: ContingencyTable) -> ContingencyResult:
    """
    Compute Tschuprow’s T statistic for association between categorical variables.

    Defined as:

        T = sqrt( χ² / ( n * sqrt((r - 1)(c - 1)) ) )

    Parameters
    ----------
    table : ContingencyTable
        A 2D contingency table.

    Returns
    -------
    ContingencyResult
        statistic : float
            Tschuprow’s T in [0, 1].
        pvalue : float
            P-value from the chi-square test.
        method : str
            Always "tschuprow_t".
        extra : dict
            Contains:
                - "n": total sample size
                - "rows": r
                - "cols": c

    Notes
    -----
    - More sensitive to table shape than Cramér’s V.
    - Undefined for tables with only one row or column.
    """
    arr = _to_array(table)
    chi = chi_square(arr)
    n = arr.sum()
    r, c = arr.shape

    denom = n * np.sqrt((r - 1) * (c - 1))
    T = np.sqrt(chi["statistic"] / denom)

    return {
        "statistic": float(T),
        "pvalue": chi["pvalue"],
        "method": "tschuprow_t",
        "extra": {"n": n, "rows": r, "cols": c},
    }
