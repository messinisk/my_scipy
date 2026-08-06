import numpy as np
import pytest

from scipy_analytics.stats.correlation import (
    classify_pearson,
    kendall_correlation,
    spearman_correlation,
)

from scipy_analytics.stats.contingency import (
    chi_square,
    odds_ratio,
    phi_coefficient,
    cramers_v,
    tschuprow_t
)

def test_pearson_classification():
    result = classify_pearson(0,85)
    assert isinstance(result, str)

def test_spearman():
    x = [1, 2, 3]
    y = [3, 2, 1]
    result = spearman_correlation(x, y)
    assert result["statistic"] < 0

def test_kendall():
    x = [1, 2, 3]
    y = [3, 2, 1]
    result = kendall_correlation(x, y)
    assert result["statistic"] < 0

def test_chi_square():
    table = [[10, 20], [20, 40]]
    result = chi_square(table)
    assert result["statistic"] == 0.0
    assert result["pvalue"] == 1.0


def test_odds_ratio():
    table = [[10, 5], [20, 10]]
    result = odds_ratio(table)
    assert result["statistic"] > 0

def test_phi():
    table = [[10, 20], [20, 40]]
    result = phi_coefficient(table)
    assert isinstance(result["statistic"], float)

def test_cramers_v():
    table = [[10, 20], [20, 40]]
    result = cramers_v(table)
    assert isinstance(result["statistic"], float)

def test_tschuprow_t():
    table = [[10, 20], [20, 40]]
    result = tschuprow_t(table)
    assert "statistic" in result
    assert result["statistic"] >= 0
