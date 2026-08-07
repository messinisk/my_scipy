import pytest
import numpy as np

from scipy_analytics.stats.descriptive import descriptive_stats


def test_descriptive_basic():
    data = [1, 2, 3, 4, 5]
    result = descriptive_stats(data)

    assert result["mean"] == 3
    assert result["median"] == 3
    assert result["min"] == 1
    assert result["max"] == 5
    assert result["count"] == 5


def test_descriptive_variance_std():
    data = [1, 2, 3, 4]
    result = descriptive_stats(data)

    assert result["variance"] == pytest.approx(1.25, rel=1e-6)
    assert result["std"] == pytest.approx(np.sqrt(1.25), rel=1e-6)




def test_descriptive_mode():
    data = [1, 2, 2, 3]
    result = descriptive_stats(data)

    assert result["mode"] == 2


def test_descriptive_skew_kurt():
    data = [1, 2, 3, 4, 100]  # skewed distribution
    result = descriptive_stats(data, kurt=True)

    assert "skewness" in result
    assert "kurtosis" in result
    assert isinstance(result["skewness"], float)
    assert isinstance(result["kurtosis"], float)


def test_descriptive_percentiles():
    data = [10, 20, 30, 40, 50]
    result = descriptive_stats(data)

    assert "percentiles" in result
    assert result["percentiles"][50] == 30
    assert result["percentiles"][0] == 10
    assert result["percentiles"][100] == 50


def test_descriptive_numpy_array():
    data = np.array([1, 2, 3])
    result = descriptive_stats(data)

    assert result["mean"] == 2
    assert result["count"] == 3


def test_descriptive_empty():
    with pytest.raises(ValueError):
        descriptive_stats([])
    
