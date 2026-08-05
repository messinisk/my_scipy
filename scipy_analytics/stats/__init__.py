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
from .distributions.scipy_wrappers import SciPyDistribution

# Correlation
from .correlation.pearson.classification import classify_pearson

# Utils
from .utils.moments import sample_moments

# QMC
from .qmc.engine import MonteCarlo

# Plots
from .plots.pdf_plot import plot_pdf

from .correlation.spearman_kendall import spearman_correlation, kendall_correlation

__all__ = [
    "SciPyDistribution",
    "classify_pearson",
    "sample_moments",
    "MonteCarlo",
    "plot_pdf",
    "spearman_correlation",
    "kendall_correlation"
]