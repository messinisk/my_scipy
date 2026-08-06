"""
Statistical hypothesis testing utilities.

Το subpackage `tests` συγκεντρώνει wrappers για τα πιο συνηθισμένα
στατιστικά tests του scipy.stats, όπως:

- t-tests (independent, paired)
- KS-test
- χ² tests
- normality tests (Shapiro, Jarque-Bera)
- variance tests (Levene)
- non-parametric tests (Mann–Whitney, Wilcoxon)
- goodness-of-fit tests
- Friedman test

Στόχος:
- Ενιαίο API για όλα τα tests
- Unified return format (statistic, pvalue, extra)
- Εύκολη αυτοματοποίηση και ενσωμάτωση σε reports
"""

from .stat_tests import (
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

__all__ = [
    "anderson_test",
    "dagostino_test",
    "fisher_test",
    "friedman_test",
    "jarque_bera_test",
    "ks_test",
    "kstest_distribution",
    "levene_test",
    "mann_whitney",
    "shapiro_test",
    "ttest_independent",
    "ttest_paired",
    "wilcoxon_test",
]
