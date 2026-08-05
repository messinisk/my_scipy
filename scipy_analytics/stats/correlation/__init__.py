"""
Correlation analysis utilities.

Περιλαμβάνει:
- Pearson correlation (classification, report, region plot)
- Spearman correlation
- Kendall correlation
"""
from .pearson import classify_pearson
from .pearson import PearsonReport
from .pearson import plot_pearson_region
from  .spearman_kendall import spearman_correlation, kendall_correlation

__all__ = [
    "classify_pearson",
    "PearsonReport",
    "plot_pearson_region",
    "spearman_correlation",
    "kendall_correlation"
]