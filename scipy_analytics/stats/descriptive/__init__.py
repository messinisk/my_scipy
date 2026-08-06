"""
Descriptive statistics utilities.

Το subpackage `descriptive` περιλαμβάνει εργαλεία για:

- summary statistics (mean, median, variance, skew, kurtosis)
- geometric & harmonic mean
- trimmed & winsorized statistics
- unified describe() interface

Στόχος:
- Ενοποιημένη περιγραφική στατιστική
- Συμβατότητα με scipy.stats.describe
- Εύκολη ενσωμάτωση σε reports και pipelines
"""

from .descriptive_stats import (
    descriptive_stats,
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

__all__ = [
    "descriptive_stats",
    "geometric_mean",
    "harmonic_mean",
    "kurtosis_value",
    "mean",
    "median",
    "mode_value",
    "skewness",
    "std",
    "trimmed_mean_value",
    "variance",
    "winsorized_mean",
]
