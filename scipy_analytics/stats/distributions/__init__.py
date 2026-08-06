"""
Probability distribution utilities.

Περιλαμβάνει:
- bases: abstract distribution interfaces
- fitting: εργαλεία MLE fitting
- scipy_wrappers: unified interface για pdf/cdf/ppf/rvs
- pdf, cdf, ppf, rvs, stats
για norm, beta, gamma, chi2, t, f, logistic, binom, poisson, geom.
"""

from .api import cdf, pdf, ppf, rvs, stats
from .bases import Distribution
from .core import get_dist
from .fitting import fit_distribution
from .scipy_wrappers import SciPyDistribution

__all__ = [
    "Distribution",
    "SciPyDistribution",
    "cdf",
    "fit_distribution",
    "get_dist",
    "pdf",
    "ppf",
    "rvs",
    "stats",
]
