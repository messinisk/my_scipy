from __future__ import annotations

from typing import Literal

import numpy as np
from sklearn.neighbors import KernelDensity

KernelType = Literal[
    "gaussian", "tophat", "epanechnikov", "exponential", "linear", "cosine"
]


class KDE:
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
        """Silverman's rule of thumb for 1D KDE."""
        x = data.ravel()
        n = len(x)

        if n < 2:
            return 1.0

        std = np.std(x, ddof=1)
        iqr = np.subtract(*np.percentile(x, [75, 25]))
        scale = min(std, iqr / 1.34)

        return float(0.9 * scale * n ** (-1/5))

    def pdf(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points)

        if points.ndim == 1:
            points = points.reshape(-1, 1)

        return np.exp(self.kde.score_samples(points)).astype(float)

    def sample(self, n: int = 1000) -> np.ndarray:
        return self.kde.sample(n)

    def grid_pdf(self, num: int = 200) -> tuple[np.ndarray, np.ndarray]:
        xmin = float(self.data.min())
        xmax = float(self.data.max())

        grid = np.linspace(xmin, xmax, num).reshape(-1, 1)
        pdf = self.pdf(grid)

        return grid, pdf
