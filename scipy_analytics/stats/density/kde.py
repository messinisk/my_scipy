"""
Unified Kernel Density Estimation (KDE) interface.

This module provides a lightweight wrapper around scikit‑learn’s
`KernelDensity` estimator, offering:

- 1D KDE estimation
- Silverman’s rule‑of‑thumb bandwidth selection
- PDF evaluation
- Sampling
- Grid evaluation for plotting

The goal is to provide a simple, NumPy‑friendly API similar to
`scipy.stats.gaussian_kde` but with scikit‑learn’s flexibility.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray
from sklearn.neighbors import KernelDensity

KernelType = Literal[
    "gaussian", "tophat", "epanechnikov", "exponential", "linear", "cosine"
]


class KDE:
    """
    Kernel Density Estimator (1D) using scikit‑learn.

    Parameters
    ----------
    data : np.ndarray
        1D array of samples. Internally reshaped to (n, 1).
    bandwidth : float or None
        KDE bandwidth. If None, Silverman's rule is used.
    kernel : {"gaussian", "tophat", "epanechnikov", "exponential", "linear", "cosine"}
        Kernel type.

    Attributes
    ----------
    data : np.ndarray
        Training data reshaped to (n, 1).
    bandwidth : float
        Selected bandwidth.
    kernel : str
        Kernel type.
    kde : KernelDensity
        Underlying scikit‑learn estimator.

    Notes
    -----
    - This class is intentionally simple and 1D‑only.
    - For multivariate KDE, use scikit‑learn directly.
    - Silverman’s rule is robust for unimodal distributions.
    """

    def __init__(
        self,
        data: np.ndarray,
        bandwidth: float | None = None,
        kernel: KernelType = "gaussian",
    ):
        data = np.asarray(data)

        # Ensure 1D → reshape to (n,1)
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        if bandwidth is None:
            bandwidth = self.silverman_bandwidth(data)

        self.data = data
        self.bandwidth = float(bandwidth)
        self.kernel = kernel

        self.kde = KernelDensity(bandwidth=self.bandwidth, kernel=self.kernel)
        self.kde.fit(self.data)

    @staticmethod
    def silverman_bandwidth(data: np.ndarray) -> float:
        """
        Silverman's rule of thumb for 1D KDE.

        Formula:
            h = 0.9 * min(std, IQR/1.34) * n^(-1/5)

        Parameters
        ----------
        data : np.ndarray
            1D or (n,1) array.

        Returns
        -------
        float
            Bandwidth estimate.

        Notes
        -----
        - Uses robust scale estimate: min(std, IQR/1.34).
        - Returns 1.0 for n < 2.
        """
        x = data.ravel()
        n = len(x)

        if n < 2:
            return 1.0

        std = np.std(x, ddof=1)
        iqr = np.subtract(*np.percentile(x, [75, 25]))
        scale = min(std, iqr / 1.34)

        return float(0.9 * scale * n ** (-1 / 5))

    def pdf(self, points: np.ndarray) -> NDArray[np.float64]:
        """
        Evaluate the KDE PDF at given points.

        Parameters
        ----------
        points : np.ndarray
            1D array of evaluation points.

        Returns
        -------
        np.ndarray
            PDF values at each point.

        Notes
        -----
        - Points are reshaped to (n,1).
        - Uses exp(score_samples).
        """
        points = np.asarray(points)

        if points.ndim == 1:
            points = points.reshape(-1, 1)

        return np.exp(self.kde.score_samples(points)).astype(float)  # type: ignore

    def sample(self, n: int = 1000) -> np.ndarray:
        """
        Draw samples from the KDE model.

        Parameters
        ----------
        n : int
            Number of samples.

        Returns
        -------
        np.ndarray
            Sampled points.
        """
        return self.kde.sample(n)  # type: ignore

    def grid_pdf(self, num: int = 200) -> tuple[np.ndarray, np.ndarray]:
        """
        Evaluate the KDE on a regular grid for plotting.

        Parameters
        ----------
        num : int
            Number of grid points.

        Returns
        -------
        (grid, pdf) : tuple[np.ndarray, np.ndarray]
            Grid points and corresponding PDF values.

        Notes
        -----
        - Grid spans [min(data), max(data)].
        """
        xmin = float(self.data.min())
        xmax = float(self.data.max())

        grid = np.linspace(xmin, xmax, num).reshape(-1, 1)
        pdf = self.pdf(grid)

        return grid, pdf
