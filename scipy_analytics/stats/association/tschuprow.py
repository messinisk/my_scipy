import numpy as np
from scipy.stats import chi2_contingency


def tschuprows_t(table: np.ndarray) -> float:
    """
    Compute Tschuprow's T statistic for measuring association between two
    categorical variables based on a contingency table.

    Tschuprow's T is a normalized chi-square measure of association that
    adjusts for table shape. It is similar to Cramér's V but uses the
    geometric mean of (r - 1) and (c - 1) instead of the minimum dimension.

    The statistic is defined as:

        T = sqrt( χ² / ( n * sqrt((r - 1)(c - 1)) ) )

    where:
        χ² : chi-square statistic from the contingency table
        n  : total number of observations
        r  : number of rows in the table
        c  : number of columns in the table

    Parameters
    ----------
    table : np.ndarray
        A 2D contingency table of non-negative counts. Must have shape
        (r, c) with r >= 2 and c >= 2. If the table is degenerate
        (i.e., r < 2 or c < 2), the function returns NaN.

    Returns
    -------
    float
        Tschuprow's T statistic in the range [0, 1].
        Returns NaN for degenerate tables or if the statistic cannot be
        computed due to zero marginal variation.

    Notes
    -----
    - Tschuprow's T is symmetric: T(X, Y) = T(Y, X).
    - Unlike Cramér's V, T is sensitive to table shape and is generally
      lower for rectangular tables.
    - The statistic is undefined for tables with only one row or one column.
    - Uses `scipy.stats.chi2_contingency` with `correction=False` to avoid
      Yates' continuity correction, which is standard for association metrics.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy_analytics.stats.association import tschuprows_t
    >>> table = np.array([[10, 20], [20, 40]])
    >>> tschuprows_t(table)
    0.0

    Degenerate input:

    >>> tschuprows_t(np.array([[5, 5, 5]]))
    nan

    """
    chi2, _, _, _ = chi2_contingency(table, correction=False)
    n = table.sum()
    r, c = table.shape

    # Degenerate table → undefined statistic
    if r < 2 or c < 2:
        return float("nan")

    value = np.sqrt(chi2 / (n * np.sqrt((r - 1) * (c - 1))))
    return float(value)
