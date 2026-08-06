from collections.abc import Sequence

import numpy as np
from numpy import ndarray
from scipy.stats import (
    gmean,
    hmean,
    kurtosis,
    mode,
    mstats,
    skew,
    trim_mean,
)

from .types import DescriptiveResult

NumericArray = ndarray | Sequence[float]


def mean(data: NumericArray) -> DescriptiveResult:
    """
    Υπολογίζει τον αριθμητικό μέσο όρο.

    Parameters
    ----------
    data : NumericArray
        Ακολουθία αριθμητικών τιμών.

    Returns
    -------
    DescriptiveResult
        Η τιμή του μέσου όρου.
    """
    value = float(np.mean(data))
    return {"stat": value, "method": "mean", "extra": {}}


def median(data: NumericArray) -> DescriptiveResult:
    """
    Υπολογίζει τη διάμεσο.

    Η διάμεσος είναι το κεντρικό σημείο της κατανομής και είναι ανθεκτική
    σε ακραίες τιμές (outliers).

    Parameters
    ----------
    data : NumericArray

    Returns
    -------
    DescriptiveResult
    """
    value = float(np.median(data))
    return {"stat": value, "method": "median", "extra": {}}


def mode_value(data: NumericArray) -> DescriptiveResult:
    """
    Υπολογίζει τη συχνότερη τιμή (mode).

    Returns
    -------
    DescriptiveResult
        extra["count"] περιέχει τον αριθμό εμφανίσεων της mode.
    """
    m = mode(data, keepdims=True)
    return {
        "stat": float(m.mode[0]),
        "method": "mode",
        "extra": {"count": int(m.count[0])},
    }


def variance(data: NumericArray, ddof: int = 1) -> DescriptiveResult:
    """
    Υπολογίζει τη δειγματική διακύμανση.

    Parameters
    ----------
    data : NumericArray
    ddof : int
        Degrees of freedom (default 1 για δειγματική διακύμανση).

    Returns
    -------
    DescriptiveResult
    """
    value = float(np.var(data, ddof=ddof))
    return {"stat": value, "method": "variance", "extra": {"ddof": ddof}}


def std(data: NumericArray, ddof: int = 1) -> DescriptiveResult:
    """
    Υπολογίζει την τυπική απόκλιση.

    Returns
    -------
    DescriptiveResult
    """
    value = float(np.std(data, ddof=ddof))
    return {"stat": value, "method": "std", "extra": {"ddof": ddof}}


def skewness(data: NumericArray) -> DescriptiveResult:
    """
    Υπολογίζει την ασυμμετρία (skewness).

    Returns
    -------
    DescriptiveResult
    """
    value = float(skew(data))
    return {"stat": value, "method": "skewness", "extra": {}}


def kurtosis_value(data: NumericArray) -> DescriptiveResult:
    """
    Υπολογίζει την κύρτωση (kurtosis).

    Returns
    -------
    DescriptiveResult
    """
    value = float(kurtosis(data))
    return {"stat": value, "method": "kurtosis", "extra": {}}


def trimmed_mean_value(
    data: NumericArray, proportion: float = 0.1
) -> DescriptiveResult:
    """
    Υπολογίζει trimmed mean.

    Αφαιρεί ένα ποσοστό από τις χαμηλές και υψηλές τιμές πριν τον υπολογισμό.

    Parameters
    ----------
    proportion : float
        Ποσοστό trimming από κάθε άκρο.

    Returns
    -------
    DescriptiveResult
    """
    value = float(trim_mean(data, proportion))
    return {
        "stat": value,
        "method": "trimmed_mean",
        "extra": {"proportion": proportion},
    }


def winsorized_mean(
    data: NumericArray, limits: tuple[float, float] = (0.1, 0.1)
) -> DescriptiveResult:
    """
    Υπολογίζει winsorized mean.

    Αντί να αφαιρεί ακραίες τιμές, τις αντικαθιστά με τιμές στα όρια.

    Parameters
    ----------
    limits : Tuple[float, float]
        Ποσοστά winsorization για χαμηλό και υψηλό άκρο.

    Returns
    -------
    DescriptiveResult
    """
    w = mstats.winsorize(data, limits=limits)
    value = float(np.mean(w))
    return {
        "stat": value,
        "method": "winsorized_mean",
        "extra": {"limits": limits},
    }


def geometric_mean(data: NumericArray) -> DescriptiveResult:
    """
    Υπολογίζει τον γεωμετρικό μέσο.

    Returns
    -------
    DescriptiveResult
    """
    value = float(gmean(data))
    return {"stat": value, "method": "geometric_mean", "extra": {}}


def harmonic_mean(data: NumericArray) -> DescriptiveResult:
    """
    Υπολογίζει τον αρμονικό μέσο.

    Returns
    -------
    DescriptiveResult
    """
    value = float(hmean(data))
    return {"stat": value, "method": "harmonic_mean", "extra": {}}
