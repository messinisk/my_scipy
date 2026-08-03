from scipy.stats import alpha
from scipy.stats import gamma
from scipy.stats import beta
from scipy.stats import norm
from scipy.stats import t

from scipy_analytics.distributions.scipy_wrappers import SciPyDistribution
from scipy_analytics.stats.pearson import classify_pearson
from scipy_analytics.utils.moments import sample_moments
from scipy_analytics.montecarlo.engine import MonteCarlo
from scipy_analytics.plotting.pdf_plot import plot_pdf

import math

def main():
    # dist = SciPyDistribution(alpha, 30.57)
    dist = SciPyDistribution(gamma, 5.0)
    # dist = SciPyDistribution(beta, 2.0, 5.0)
    # dist = SciPyDistribution(norm, 0, 1)
    # dist = SciPyDistribution(t, 10)

    mean, var, skew, kurt = dist.stats()
    print("Moments:", mean, var, skew, kurt)

    # Fallback για infinite moments
    if math.isinf(mean) or math.isinf(var):
        print("Distribution has infinite mean/variance — Pearson classification not possible.")
        return

    ptype = classify_pearson(skew, kurt)
    print("Pearson Type:", ptype)

    mc = MonteCarlo(dist)
    samples = mc.simulate(10_000)

    s_mean, s_var, s_skew, s_kurt = sample_moments(samples)
    print("Sample Moments:", s_mean, s_var, s_skew, s_kurt)

    plot_pdf(dist, 0, 10)


if __name__ == "__main__":
    main()
