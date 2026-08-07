"""
Bandwidth selection utilities for Kernel Density Estimation (KDE).

This module provides cross‑validated bandwidth selection using scikit‑learn’s
`GridSearchCV` and `KernelDensity`. Bandwidth selection is critical for KDE
performance: too small → overfitting, too large → oversmoothing.

Functions
---------
cv_bandwidth(data, bandwidths=None)
    Select the optimal bandwidth via 5‑fold cross‑validation.

Notes
-----
- Uses scikit‑learn’s KernelDensity (Gaussian kernel).
- Suitable for 1D data.
- Returns a scalar bandwidth.
"""

import numpy as np
from sklearn.model_selection import GridSearchCV

from scipy_analytics.stats.information.mutual_information import KernelDensity


def cv_bandwidth(data: np.ndarray, bandwidths=None) -> float:
    """
    Select the optimal KDE bandwidth using cross‑validation.

    Parameters
    ----------
    data : np.ndarray
        1D array of samples.
    bandwidths : array-like or None
        Candidate bandwidths. If None, uses logspace from 1e‑2 to 10.

    Returns
    -------
    float
        The best bandwidth according to 5‑fold cross‑validation.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy_analytics.stats.density.bandwidth import cv_bandwidth
    >>> data = np.random.randn(500)
    >>> bw = cv_bandwidth(data)
    >>> bw
    0.42

    Notes
    -----
    - Uses Gaussian kernel.
    - Uses scikit‑learn’s GridSearchCV.
    - Data is reshaped to (n, 1) internally.
    """
    if bandwidths is None:
        bandwidths = np.logspace(-2, 1, 20)

    grid = GridSearchCV(
        KernelDensity(kernel="gaussian"),
        {"bandwidth": bandwidths},
        cv=5,
    )
    grid.fit(data.reshape(-1, 1))
    return float(grid.best_params_["bandwidth"])
