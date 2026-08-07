"""
Mutual Information (MI) estimation utilities.

This module provides functions for computing mutual information between
variables, both discrete and continuous.

Definitions
-----------
Mutual Information measures dependence between variables:

    MI(X, Y) = ∑_x ∑_y p(x, y) log( p(x, y) / (p(x)p(y)) )

For continuous variables, MI is estimated using Kernel Density Estimation (KDE):

    MI(X, Y) = E[ log p(x, y) - log p(x) - log p(y) ]

Functions
---------
kde_density(data, bw)
    Fit a Gaussian KDE using scikit‑learn.

mutual_information_discrete(x, y, bins)
    Histogram‑based MI for discrete or binned continuous variables.

mutual_information_continuous(x, y, bw, samples)
    KDE‑based MI estimator for continuous variables.

Notes
-----
- Continuous MI estimation uses Monte Carlo sampling.
- Degenerate inputs (constant variables) return NaN.
- KDE estimation is sensitive to bandwidth selection.
"""

import numpy as np
from sklearn.neighbors import KernelDensity


def kde_density(data: np.ndarray, bw: float = 0.3) -> KernelDensity:
    """
    Fit a Gaussian Kernel Density Estimator (KDE).

    Parameters
    ----------
    data : np.ndarray
        Input data. Must be shaped as (n, d).
    bw : float
        Bandwidth for KDE.

    Returns
    -------
    KernelDensity
        Fitted KDE model.

    Examples
    --------
    >>> kde = kde_density(np.random.randn(100, 1), bw=0.2)
    >>> kde.score_samples([[0.0]])
    array([-1.23])
    """
    kde = KernelDensity(bandwidth=bw, kernel="gaussian")
    kde.fit(data)
    return kde


def mutual_information_discrete(x, y, bins=20):
    """
    Compute mutual information for discrete variables (or binned continuous data).

    Parameters
    ----------
    x, y : array-like
        Input variables.
    bins : int
        Number of histogram bins.

    Returns
    -------
    float
        Estimated mutual information.

    Notes
    -----
    - Uses histogram2d to estimate joint distribution.
    - Suitable for discrete variables or coarse continuous MI.
    - Zero-probability bins are ignored.

    Examples
    --------
    >>> x = np.random.randint(0, 5, 1000)
    >>> y = np.random.randint(0, 5, 1000)
    >>> mutual_information_discrete(x, y)
    0.01
    """
    joint_hist, _, _ = np.histogram2d(x, y, bins=bins)
    joint_prob = joint_hist / joint_hist.sum()

    px = joint_prob.sum(axis=1)
    py = joint_prob.sum(axis=0)

    mi = 0.0
    for i in range(len(px)):
        for j in range(len(py)):
            if joint_prob[i, j] > 0:
                mi += joint_prob[i, j] * np.log(joint_prob[i, j] / (px[i] * py[j]))
    return float(mi)


def mutual_information_continuous(x, y, bw=0.3, samples=2000):
    """
    Estimate mutual information for continuous variables using KDE.

    Parameters
    ----------
    x, y : array-like
        Continuous variables.
    bw : float
        KDE bandwidth.
    samples : int
        Number of Monte Carlo samples.

    Returns
    -------
    float
        Estimated mutual information.

    Notes
    -----
    - Degenerate inputs (constant variables) return NaN.
    - KDE estimation uses scikit‑learn's KernelDensity.
    - MI is computed as:
          MI = E[ log p(x, y) - log p(x) - log p(y) ]
    - Sampling is performed from the empirical joint distribution.

    Examples
    --------
    >>> x = np.random.randn(1000)
    >>> y = x + 0.1*np.random.randn(1000)
    >>> mutual_information_continuous(x, y)
    0.45
    """
    # Degenerate input → MI undefined
    if np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan")

    xy = np.vstack([x, y]).T

    kde_xy = kde_density(xy, bw)
    kde_x = kde_density(x.reshape(-1, 1), bw)
    kde_y = kde_density(y.reshape(-1, 1), bw)

    pts = xy[np.random.choice(len(x), samples)]
    log_pxy = kde_xy.score_samples(pts)
    log_px = kde_x.score_samples(pts[:, [0]])
    log_py = kde_y.score_samples(pts[:, [1]])

    return float(np.mean(log_pxy - log_px - log_py))
