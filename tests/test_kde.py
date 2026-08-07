import numpy as np
import pytest

from scipy_analytics.stats.density.kde import KDE





def test_kde_pdf_integrates():
    x = np.random.randn(1000)
    kde = KDE(x)

    grid, pdf = kde.grid_pdf(2000)
    dx = grid[1] - grid[0]

    assert np.sum(pdf * dx) == pytest.approx(1.0, abs=0.05)

def test_kde_normal():
    x = np.random.normal(0, 1, 2000)
    kde = KDE(x)

    grid, pdf = kde.grid_pdf(2000)

    # flatten grid for true PDF
    grid_flat = grid.ravel()

    true_pdf = 1/np.sqrt(2*np.pi) * np.exp(-0.5 * grid_flat**2)

    assert np.mean(np.abs(pdf - true_pdf)) < 0.05

