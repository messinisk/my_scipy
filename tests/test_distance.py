from scipy_analytics.stats.correlation.distance import distance_correlation
import numpy as np
import pytest


def test_distance_correlation_independent():
    np.random.seed(0)
    x = np.random.randn(500)
    y = np.random.randn(500)

    d = distance_correlation(x, y)

    assert d == pytest.approx(0.0, abs=0.1)



def test_distance_correlation_linear():
    np.random.seed(0)
    x = np.random.randn(500)
    y = 4*x + 1

    d = distance_correlation(x, y)

    assert d == pytest.approx(1.0, abs=0.01)

def test_distance_correlation_nonlinear():
    np.random.seed(0)
    x = np.random.uniform(-3, 3, 500)
    y = np.sin(x)

    d = distance_correlation(x, y)

    assert d > 0.8

def test_distance_correlation_constant():
    x = np.ones(100)
    y = np.random.randn(100)

    d = distance_correlation(x, y)

    assert np.isnan(d)
