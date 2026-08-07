"""
Distance correlation and related utilities.

This module implements distance covariance, distance variance, and
distance correlation as introduced by Székely, Rizzo & Bakirov (2007).
Distance correlation is a powerful measure of dependence that detects
nonlinear associations where Pearson correlation fails.

Definitions
-----------
Given random variables X and Y with samples x_i and y_i:

1. Compute pairwise distance matrices:
       a_ij = |x_i - x_j|
       b_ij = |y_i - y_j|

2. Double-center each distance matrix:
       A_ij = a_ij - row_mean_i - col_mean_j + total_mean
       B_ij = b_ij - row_mean_i - col_mean_j + total_mean

3. Distance covariance:
       dCov(X, Y) = sqrt( mean(A_ij * B_ij) )

4. Distance variance:
       dVar(X) = sqrt( mean(A_ij * A_ij) )

5. Distance correlation:
       dCor(X, Y) = dCov(X, Y) / sqrt( dVar(X) * dVar(Y) )

Distance correlation satisfies:
- dCor(X, Y) = 0  ⇔  X and Y are independent
- 0 ≤ dCor ≤ 1

All functions operate on 1D NumPy arrays.

References
----------
Székely, G. J., Rizzo, M. L., & Bakirov, N. K. (2007).
"Measuring and testing dependence by correlation of distances."
Annals of Statistics, 35(6), 2769–2794.
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform

# ---------------------------------------------------------------------------
# Distance matrix
# ---------------------------------------------------------------------------


def distance_matrix(x: np.ndarray) -> np.ndarray:
    """
    Compute the pairwise Euclidean distance matrix for a 1D array.

    Parameters
    ----------
    x : np.ndarray
        1D array of numeric values.

    Returns
    -------
    np.ndarray
        A square matrix D where D[i, j] = |x[i] - x[j]|.
    """
    return squareform(pdist(x.reshape(-1, 1)))


# ---------------------------------------------------------------------------
# Double-centering
# ---------------------------------------------------------------------------


def double_center(D: np.ndarray) -> np.ndarray:
    """
    Apply double-centering to a distance matrix.

    For a distance matrix D, the double-centered form is:

        A_ij = D_ij - row_mean_i - col_mean_j + total_mean

    Parameters
    ----------
    D : np.ndarray
        A square distance matrix.

    Returns
    -------
    np.ndarray
        The double-centered matrix.
    """
    row_mean = D.mean(axis=1, keepdims=True)
    col_mean = D.mean(axis=0, keepdims=True)
    total_mean = D.mean()
    return D - row_mean - col_mean + total_mean


# ---------------------------------------------------------------------------
# Distance covariance
# ---------------------------------------------------------------------------


def distance_covariance(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute the distance covariance between two 1D arrays.

    Parameters
    ----------
    x, y : np.ndarray
        1D arrays of equal length.

    Returns
    -------
    float
        The distance covariance dCov(X, Y).
    """
    A = double_center(distance_matrix(x))
    B = double_center(distance_matrix(y))
    return float(np.sqrt(np.mean(A * B)))


# ---------------------------------------------------------------------------
# Distance variance
# ---------------------------------------------------------------------------


def distance_variance(x: np.ndarray) -> float:
    """
    Compute the distance variance of a 1D array.

    Parameters
    ----------
    x : np.ndarray
        1D array.

    Returns
    -------
    float
        The distance variance dVar(X).
    """
    A = double_center(distance_matrix(x))
    return float(np.sqrt(np.mean(A * A)))


# ---------------------------------------------------------------------------
# Distance correlation
# ---------------------------------------------------------------------------


def distance_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute the distance correlation between two 1D arrays.

    Distance correlation detects nonlinear dependence and is zero if and
    only if the variables are independent.

    Parameters
    ----------
    x, y : np.ndarray
        1D arrays of equal length.

    Returns
    -------
    float
        The distance correlation dCor(X, Y) in [0, 1].
        Returns NaN if either variable has zero distance variance
        (i.e., constant input).

    Notes
    -----
    - dCor(X, Y) = 0  ⇔  X and Y are independent.
    - dCor(X, Y) = 1  for perfect linear or nonlinear dependence.
    - If x or y is constant, distance variance is zero → result is NaN.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy_analytics.stats.correlation.distance import distance_correlation
    >>> x = np.random.randn(100)
    >>> y = x ** 2
    >>> distance_correlation(x, y)
    0.8  # strong nonlinear dependence

    Degenerate input:

    >>> distance_correlation(np.ones(10), np.arange(10))
    nan
    """
    dcov = distance_covariance(x, y)
    dvar_x = distance_variance(x)
    dvar_y = distance_variance(y)

    # Degenerate input → undefined
    if dvar_x == 0 or dvar_y == 0:
        return float("nan")

    return float(dcov / np.sqrt(dvar_x * dvar_y))
