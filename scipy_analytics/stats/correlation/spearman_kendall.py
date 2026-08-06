"""
Spearman & Kendall correlation utilities.

Αυτό το module παρέχει ενιαία wrappers για:
- Spearman rank correlation
- Kendall tau correlation

Όλες οι συναρτήσεις επιστρέφουν unified dict format:
{
    "coef": float,
    "pvalue": float,
    "method": str
}

Στόχος:
- Ενοποιημένη συσχέτιση κατάταξης
- Plug-and-play χρήση σε reports, pipelines και plotting
- Συμβατότητα με scipy.stats.spearmanr και scipy.stats.kendalltau
"""

from scipy.stats import kendalltau, spearmanr


def spearman_correlation(x, y):
    """
    Υπολογισμός Spearman rank correlation.

    Parameters
    ----------
    - x : array-like
        Πρώτη μεταβλητή.
    - y : array-like
        Δεύτερη μεταβλητή.

    Returns
    -------
    - Dict {
        "statistic": float(stat),
        "pvalue": float(p),
        "method": "spearman",
        "extra": {}
    }
    """
    stat, p = spearmanr(x, y)
    return {
        "statistic": float(stat),
        "pvalue": float(p),
        "method": "spearman",
        "extra": {},
    }


def kendall_correlation(x, y):
    """
    Υπολογισμός Kendall tau correlation.

    Parameters
    ----------
    x : array-like
        Πρώτη μεταβλητή.
    y : array-like
        Δεύτερη μεταβλητή.

    Returns
    -------
    dict
        {
        "statistic": float(stat),
        "pvalue": float(p),
        "method": "kendall",
        "extra": {}
    }
    """
    stat, p = kendalltau(x, y)
    return {
        "statistic": float(stat),
        "pvalue": float(p),
        "method": "kendall",
        "extra": {},
    }
