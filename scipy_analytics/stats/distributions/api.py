# scipy_analytics/stats/distributions/api.py

from typing import Any

import numpy as np

from .core import get_dist


def pdf(name: str, x: Any, **params: Any) -> np.ndarray:
    dist = get_dist(name)(**params)
    return dist.pdf(x)


def cdf(name: str, x: Any, **params: Any) -> np.ndarray:
    dist = get_dist(name)(**params)
    return dist.cdf(x)


def ppf(name: str, q: Any, **params: Any) -> np.ndarray:
    dist = get_dist(name)(**params)
    return dist.ppf(q)


def rvs(name: str, size: int = 1, **params: Any) -> np.ndarray:
    dist = get_dist(name)(**params)
    return dist.rvs(size=size)


def stats(name: str, **params: Any) -> dict[str, float]:
    dist = get_dist(name)(**params)
    mean, var, skew, kurt = dist.stats(moments="mvsk")
    return {
        "mean": float(mean),
        "var": float(var),
        "skew": float(skew),
        "kurtosis": float(kurt),
    }
