"""
Unified statistical API.

Το subpackage `stats` συγκεντρώνει όλες τις στατιστικές λειτουργίες
του scipy_analytics σε ενιαίο API, οργανωμένο σε θεματικές ενότητες:

- correlation
- descriptive
- tests
- distributions
- kde
- qmc
- contingency
- plots
- utils
"""

# Distributions
from .contingency.contingency import (
    chi_square,
    cramers_v,
    odds_ratio,
    phi_coefficient,
)

# Correlation
from .correlation.pearson.classification import classify_pearson
from .correlation.spearman_kendall import kendall_correlation, spearman_correlation
from .descriptive.descriptive_stats import (
    geometric_mean,
    harmonic_mean,
    kurtosis_value,
    mean,
    median,
    mode_value,
    skewness,
    std,
    trimmed_mean_value,
    variance,
    winsorized_mean,
)
from .distributions import cdf, pdf, ppf, rvs
from .distributions import stats as dist_stats
from .distributions.scipy_wrappers import SciPyDistribution
from .fitting import fit_distribution, summarize_fit

# Plots
from .plots.pdf_plot import plot_pdf

# QMC
from .qmc.engine import (
    MonteCarlo,
    halton_sample,
    lhs_sample,
    qmc_distribution_sample,
    random_sample,
    sobol_sample,
)
from .tests.stat_tests import (
    anderson_test,
    dagostino_test,
    fisher_test,
    friedman_test,
    jarque_bera_test,
    ks_test,
    kstest_distribution,
    levene_test,
    mann_whitney,
    shapiro_test,
    ttest_independent,
    ttest_paired,
    wilcoxon_test,
)

# Utils
from .utils.moments import sample_moments

__all__ = [
    "MonteCarlo",
    "SciPyDistribution",
    "anderson_test",
    "cdf",
    "chi_square",
    "classify_pearson",
    "cramers_v",
    "dagostino_test",
    "dist_stats",
    "fisher_test",
    "fit_distribution",
    "friedman_test",
    "geometric_mean",
    "halton_sample",
    "harmonic_mean",
    "jarque_bera_test",
    "kendall_correlation",
    "ks_test",
    "kstest_distribution",
    "kurtosis_value",
    "levene_test",
    "lhs_sample",
    "mann_whitney",
    "mean",
    "median",
    "mode_value",
    "odds_ratio",
    "pdf",
    "phi_coefficient",
    "plot_pdf",
    "ppf",
    "qmc_distribution_sample",
    "random_sample",
    "rvs",
    "sample_moments",
    "shapiro_test",
    "skewness",
    "sobol_sample",
    "spearman_correlation",
    "std",
    "summarize_fit",
    "trimmed_mean_value",
    "ttest_independent",
    "ttest_paired",
    "variance",
    "wilcoxon_test",
    "winsorized_mean",
]
