import numpy as np
import pytest

from scipy_analytics.stats.density.kde import KDE


def test_kde_bandwidth_silverman():
    x = np.random.randn(1000)
    bw = KDE.silverman_bandwidth(x.reshape(-1, 1))
    assert bw > 0

def test_kde_sampling():
    x = np.random.randn(1000)
    kde = KDE(x)
    samples = kde.sample(100)
    assert samples.shape == (100, 1)
