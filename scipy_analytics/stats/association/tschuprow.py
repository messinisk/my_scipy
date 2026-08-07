import numpy as np
from scipy.stats import chi2_contingency


def tschuprows_t(table: np.ndarray) -> float:
    chi2, _, _, _ = chi2_contingency(table, correction=False)
    n = table.sum()
    r, c = table.shape
    if r < 2 or c < 2:
        return float("nan")
    return np.sqrt(chi2 / (n * np.sqrt((r - 1) * (c - 1))))
