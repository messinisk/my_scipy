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
