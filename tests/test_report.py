import pytest
from scipy_analytics.stats.correlation.pearson.report import PearsonReport

def test_report_basic():
    rep = PearsonReport(0.75)
    s = rep.summary()

    assert s["r"] == 0.75
    assert s["direction"] == "positive"
    assert s["strength"] == "strong"
    assert "pearson_type" in s

def test_report_negative():
    rep = PearsonReport(-0.3)
    s = rep.summary()

    assert s["direction"] == "negative"
    assert s["strength"] == "weak"

def test_report_zero():
    rep = PearsonReport(0.0)
    s = rep.summary()

    assert s["direction"] == "none"
    assert s["strength"] == "very weak"

def test_report_invalid_type():
    with pytest.raises(TypeError):
        PearsonReport("abc") # type: ignore[arg-type]

def test_report_invalid_range():
    with pytest.raises(ValueError):
        PearsonReport(2.0)
