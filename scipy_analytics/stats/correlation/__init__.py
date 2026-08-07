"""
Correlation analysis utilities.

Περιλαμβάνει:
- Pearson correlation (classification, report, region plot)
- Spearman correlation
- Kendall correlation
"""

from .distance import distance_correlation, distance_covariance, distance_variance
from .pearson import PearsonReport, classify_pearson, plot_pearson_region
from .spearman_kendall import kendall_correlation, spearman_correlation

__all__ = [
    "PearsonReport",
    "classify_pearson",
    "distance_correlation",
    "distance_covariance",
    "distance_variance",
    "kendall_correlation",
    "plot_pearson_region",
    "spearman_correlation",
]
