"""
Correlation analysis utilities.

Περιλαμβάνει:
- Pearson correlation (classification, report, region plot)
- Spearman correlation
- Kendall correlation
"""

from .pearson import PearsonReport, classify_pearson, plot_pearson_region
from .spearman_kendall import kendall_correlation, spearman_correlation

__all__ = [
    "PearsonReport",
    "classify_pearson",
    "kendall_correlation",
    "plot_pearson_region",
    "spearman_correlation",
]
