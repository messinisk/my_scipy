import numpy as np
import pytest

from scipy_analytics.stats.association import tschuprows_t


def test_tschuprows_t_basic():
    table = np.array([[10, 5],
                      [3, 12]])

    t = tschuprows_t(table)

    assert pytest.approx(t, rel=1e-6) == 0.470871  # known value



def test_tschuprows_t_independent():
    table = np.array([[10, 10],
                      [10, 10]])

    t = tschuprows_t(table)

    assert t == pytest.approx(0.0, abs=1e-12)


def test_tschuprows_t_perfect():
    table = np.array([[10, 0],
                      [0, 10]])

    t = tschuprows_t(table)

    assert t == pytest.approx(1.0, abs=1e-12)



def test_tschuprows_t_degenerate():
    table = np.array([[1, 2, 3]])

    t = tschuprows_t(table)

    assert np.isnan(t)
