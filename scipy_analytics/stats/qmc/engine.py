"""
Quasi-Monte Carlo sampling utilities.

Υποστηρίζει:
- Sobol
- Halton
- Latin Hypercube
- Random sampling
- Sampling από distributions μέσω QMC

Κάθε συνάρτηση επιστρέφει SampleResult:
{
    "samples": np.ndarray,
    "method": str,
    "extra": dict
}
"""

from collections.abc import Sequence

import numpy as np
from numpy import ndarray
from scipy.stats import qmc, rv_continuous, rv_discrete
from scipy.stats._distn_infrastructure import rv_frozen

from .types import SampleResult

NumericArray = ndarray | Sequence[float]

DistributionType = rv_continuous | rv_discrete | rv_frozen


class MonteCarlo:
    """
    Κινητήρας Monte Carlo για προσομοιώσεις με βάση
    οποιαδήποτε συνεχής κατανομή.

    Η κλάση αυτή υλοποιεί την θεωρητική διαδικασία
    Monte Carlo: δειγματοληψία από μια κατανομή και
    υπολογισμό στατιστικών ή λειτουργικών μεγεθών.

    Parameters
    ----------
    distribution : Distribution
        Αντικείμενο κατανομής που ακολουθεί το ενιαίο API.

    Methods
    -------
    simulate(n)
        Παράγει n τυχαίες τιμές από την κατανομή.

    experiment(n, func)
        Εκτελεί πείραμα Monte Carlo εφαρμόζοντας μια
        συνάρτηση func στα παραγόμενα δείγματα.

    """

    def __init__(self, distribution):
        self.dist = distribution

    def simulate(self, n):
        """
        simulate(n)
        Παράγει n τυχαίες τιμές από την κατανομή.
        """
        return self.dist.rvs(size=n)

    def experiment(self, n, func):
        """
        Εκτελεί πείραμα Monte Carlo εφαρμόζοντας μια
        συνάρτηση func στα παραγόμενα δείγματα.
        """
        samples = self.simulate(n)
        return func(samples)


def sobol_sample(dim: int, n: int, scramble: bool = True) -> SampleResult:
    sampler = qmc.Sobol(d=dim, scramble=scramble)
    samples = sampler.random(n)
    return {
        "samples": samples,
        "method": "sobol",
        "extra": {"dim": dim, "n": n, "scramble": scramble},
    }


def halton_sample(dim: int, n: int, scramble: bool = True) -> SampleResult:
    sampler = qmc.Halton(d=dim, scramble=scramble)
    samples = sampler.random(n)
    return {
        "samples": samples,
        "method": "halton",
        "extra": {"dim": dim, "n": n, "scramble": scramble},
    }


def lhs_sample(dim: int, n: int) -> SampleResult:
    sampler = qmc.LatinHypercube(d=dim)
    samples = sampler.random(n)
    return {
        "samples": samples,
        "method": "lhs",
        "extra": {"dim": dim, "n": n},
    }


def random_sample(dim: int, n: int) -> SampleResult:
    samples = np.random.rand(n, dim)
    return {
        "samples": samples,
        "method": "random",
        "extra": {"dim": dim, "n": n},
    }


def qmc_distribution_sample(
    dist: DistributionType,
    dim: int,
    n: int,
    method: str = "sobol",
) -> SampleResult:
    if method == "sobol":
        base = sobol_sample(dim, n)["samples"]
    elif method == "halton":
        base = halton_sample(dim, n)["samples"]
    elif method == "lhs":
        base = lhs_sample(dim, n)["samples"]
    elif method == "random":
        base = random_sample(dim, n)["samples"]
    else:
        raise ValueError(f"Unknown QMC method: {method}")

    mapped = dist.ppf(base)

    # Αντί για dist.dist.name / dist.name → ασφαλές, τυπικά χρήσιμο:
    dist_name = dist.__class__.__name__

    return {
        "samples": mapped,
        "method": f"qmc_{method}",
        "extra": {"dim": dim, "n": n, "distribution": dist_name},
    }
