"""
Core distribution registry for the `distributions` API.

This module defines a mapping from string names to SciPy distribution
objects and provides the `get_dist` function, which retrieves the
corresponding distribution class.

It acts as the central registry for all supported probability
distributions in the `scipy_analytics.stats.distributions` subpackage.

Features
--------
- Name‑based lookup for SciPy distributions
- Unified interface for pdf/cdf/ppf/rvs/stats via `api.py`
- Easy extensibility: add new distributions by updating `DistributionMap`

Supported distributions
-----------------------
The following SciPy distributions are registered:

- norm        : Normal distribution
- beta        : Beta distribution
- gamma       : Gamma distribution
- chi2        : Chi‑square distribution
- t           : Student's t distribution
- f           : F distribution
- logistic    : Logistic distribution
- binom       : Binomial distribution
- poisson     : Poisson distribution
- geom        : Geometric distribution

Notes
-----
- All registered objects are SciPy frozen distribution classes.
- `get_dist(name)` returns the *class*, not an instance.
- Instantiation is handled by `api.py` (e.g., `pdf("norm", x, loc=0, scale=1)`).
"""

from typing import Any

from scipy.stats import (
    beta,
    binom,
    chi2,
    f,
    gamma,
    geom,
    logistic,
    norm,
    poisson,
    t,
)

# ---------------------------------------------------------------------------
# Distribution registry
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Lookup function
# ---------------------------------------------------------------------------


def get_dist(name: str) -> Any:
    """
    Retrieve a SciPy distribution class by name.

    Parameters
    ----------
    name : str
        Name of the distribution (e.g., "norm", "beta", "gamma").

    Returns
    -------
    Any
        A SciPy distribution class (not an instance).

    Raises
    ------
    ValueError
        If the distribution name is not registered.

    Examples
    --------
    >>> dist = get_dist("norm")
    >>> dist(loc=0, scale=1).pdf(0)
    0.3989422804014327

    Notes
    -----
    - Instantiation is done in `api.py`.
    - Extend the registry by adding new entries to `DistributionMap`.
    """
    if name not in DistributionMap:
        raise ValueError(f"Unknown distribution: {name}")
    return DistributionMap[name]
