# scipy_analytics/stats/distributions/core.py

from typing import Any

from scipy.stats import beta, binom, chi2, f, gamma, geom, logistic, norm, poisson, t

DistributionMap: dict[str, Any] = {
    "norm": norm,
    "beta": beta,
    "gamma": gamma,
    "chi2": chi2,
    "t": t,
    "f": f,
    "logistic": logistic,
    "binom": binom,
    "poisson": poisson,
    "geom": geom,
}


def get_dist(name: str) -> Any:
    if name not in DistributionMap:
        raise ValueError(f"Unknown distribution: {name}")
    return DistributionMap[name]
