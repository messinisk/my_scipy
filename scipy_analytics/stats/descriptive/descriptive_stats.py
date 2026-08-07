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

from .types import DescriptiveResult, LowLevelResult

NumericArray = ndarray | Sequence[float]


def mean(data: NumericArray) -> LowLevelResult:
    """
    Υπολογίζει τον αριθμητικό μέσο όρο.

    Parameters
    ----------
    data : NumericArray
        Ακολουθία αριθμητικών τιμών.

    Returns
    -------
    LowLevelResult
        Η τιμή του μέσου όρου.
    """
    value = float(np.mean(data))
    return {"stat": value, "method": "mean", "extra": {}}


def median(data: NumericArray) -> LowLevelResult:
    """
    Υπολογίζει τη διάμεσο.

    Η διάμεσος είναι το κεντρικό σημείο της κατανομής και είναι ανθεκτική
    σε ακραίες τιμές (outliers).

    Parameters
    ----------
    data : NumericArray

    Returns
    -------
    LowLevelResult
    """
    value = float(np.median(data))
    return {"stat": value, "method": "median", "extra": {}}


def mode_value(data: NumericArray) -> LowLevelResult:
    """
    Υπολογίζει τη συχνότερη τιμή (mode).

    Returns
    -------
    LowLevelResult
        extra["count"] περιέχει τον αριθμό εμφανίσεων της mode.
    """
    m = mode(data, keepdims=True)
    return {
        "stat": float(m.mode[0]),
        "method": "mode",
        "extra": {"count": int(m.count[0])},
    }


def variance(data: NumericArray, ddof: int = 1) -> LowLevelResult:
    """
    Υπολογίζει τη δειγματική διακύμανση.

    Parameters
    ----------
    data : NumericArray
    ddof : int
        Degrees of freedom (default 1 για δειγματική διακύμανση).

    Returns
    -------
    LowLevelResult
    """
    value = float(np.var(data))
    return {"stat": value, "method": "variance", "extra": {"ddof": ddof}}


def std(data: NumericArray, ddof: int = 1) -> LowLevelResult:
    """
    Υπολογίζει την τυπική απόκλιση.

    Returns
    -------
    LowLevelResult
    """
    value = float(np.std(data))
    return {"stat": value, "method": "std", "extra": {"ddof": ddof}}


def skewness(data: NumericArray) -> LowLevelResult:
    """
    Υπολογίζει την ασυμμετρία (skewness).

    Returns
    -------
    LowLevelResult
    """
    value = float(skew(data))
    return {"stat": value, "method": "skewness", "extra": {}}


def kurtosis_value(data: NumericArray) -> LowLevelResult:
    """
    Υπολογίζει την κύρτωση (kurtosis).

    Returns
    -------
    LowLevelResult
    """
    value = float(kurtosis(data))
    return {"stat": value, "method": "kurtosis", "extra": {}}


def trimmed_mean_value(data: NumericArray, proportion: float = 0.1) -> LowLevelResult:
    """
    Υπολογίζει trimmed mean.

    Αφαιρεί ένα ποσοστό από τις χαμηλές και υψηλές τιμές πριν τον υπολογισμό.

    Parameters
    ----------
    proportion : float
        Ποσοστό trimming από κάθε άκρο.

    Returns
    -------
    LowLevelResult
    """
    value = float(trim_mean(data, proportion))
    return {
        "stat": value,
        "method": "trimmed_mean",
        "extra": {"proportion": proportion},
    }


def winsorized_mean(
    data: NumericArray, limits: tuple[float, float] = (0.1, 0.1)
) -> LowLevelResult:
    """
    Υπολογίζει winsorized mean.

    Αντί να αφαιρεί ακραίες τιμές, τις αντικαθιστά με τιμές στα όρια.

    Parameters
    ----------
    limits : Tuple[float, float]
        Ποσοστά winsorization για χαμηλό και υψηλό άκρο.

    Returns
    -------
    LowLevelResult
    """
    w = mstats.winsorize(data, limits=limits)
    value = float(np.mean(w))
    return {
        "stat": value,
        "method": "winsorized_mean",
        "extra": {"limits": limits},
    }


def geometric_mean(data: NumericArray) -> LowLevelResult:
    """
    Υπολογίζει τον γεωμετρικό μέσο.

    Returns
    -------
    LowLevelResult
    """
    value = float(gmean(data))
    return {"stat": value, "method": "geometric_mean", "extra": {}}


def harmonic_mean(data: NumericArray) -> LowLevelResult:
    """
    Υπολογίζει τον αρμονικό μέσο.

    Returns
    -------
    LowLevelResult
    """
    value = float(hmean(data))
    return {"stat": value, "method": "harmonic_mean", "extra": {}}


def descriptive_stats(data: NumericArray, kurt: bool = False) -> DescriptiveResult:
    arr = np.asarray(data, dtype=float)
    if arr.size == 0:
        raise ValueError("Empty input")
    return {
        "mean": mean(arr)["stat"],
        "median": median(arr)["stat"],
        "mode": mode_value(arr)["stat"],
        "variance": variance(arr)["stat"],  # ddof=1
        "std": std(arr)["stat"],  # ddof=1
        "skewness": skewness(arr)["stat"],
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "count": int(arr.size),
        "percentiles": {
            0: float(np.percentile(arr, 0)),
            25: float(np.percentile(arr, 25)),
            50: float(np.percentile(arr, 50)),
            75: float(np.percentile(arr, 75)),
            100: float(np.percentile(arr, 100)),
        },
        "kurtosis": kurtosis_value(arr)["stat"] if kurt else None,
    }
