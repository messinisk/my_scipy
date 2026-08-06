"""
Contingency table analysis utilities.

Το subpackage `contingency` παρέχει εργαλεία για:

- χ² test of independence
- odds ratio
- Cramér’s V
- φ-coefficient
- unified contingency table interface

Στόχος:
- Ενοποιημένη ανάλυση πινάκων συχνοτήτων
- Plug-and-play χρήση σε statistical workflows
- Συμβατότητα με scipy.stats.chi2_contingency
"""

from .contingency import (
    chi_square,
    cramers_v,
    odds_ratio,
    phi_coefficient,
    tschuprow_t,
)

__all__ = [
    "chi_square",
    "cramers_v",
    "odds_ratio",
    "phi_coefficient",
    "tschuprow_t",
]
