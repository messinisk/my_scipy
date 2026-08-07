"""
Mutual Information (MI) utilities.

This subpackage provides tools for estimating mutual information between
variables, both discrete and continuous. It includes:

- kde_density: KDE estimator for continuous MI
- mutual_information_discrete: histogram‑based MI for discrete variables
- mutual_information_continuous: KDE‑based MI for continuous variables

The API is designed to integrate with the `density` and `correlation`
subpackages, providing a unified interface for information‑theoretic
statistics.
"""

from .mutual_information import (
    kde_density,
    mutual_information_continuous,
    mutual_information_discrete,
)

__all__ = [
    "kde_density",
    "mutual_information_continuous",
    "mutual_information_discrete",
]
