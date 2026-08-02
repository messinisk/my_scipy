from scipy.stats import alpha

from scipy_analytics.distributions.scipy_wrappers import SciPyDistribution
from scipy_analytics.distributions.pearson import classify_pearson
from scipy_analytics.utils.moments import sample_moments
from scipy_analytics.montecarlo.engine import MonteCarlo
from scipy_analytics.plotting.pdf_plot import plot_pdf

import math

def main():
    dist = SciPyDistribution(alpha, 3.57)

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
