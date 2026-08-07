"""
Pearson System classification utilities.

This module provides tools for classifying distributions according to the
Pearson System of Curves based on skewness and kurtosis. The Pearson system
categorizes distributions into types (I–VII, Normal) depending on the values
of the standardized moments:

    β₁ = skew²
    β₂ = kurtosis
    D  = β₂ − β₁ − 1

The Pearson system is historically important because it provides a unified
framework for describing a wide variety of continuous distributions using
a differential equation. In modern practice, it is used to:

- classify empirical distributions based on sample skewness/kurtosis
- select appropriate theoretical distributions for modeling
- understand tail behavior and asymmetry

This module includes:

- PearsonType: enumeration of Pearson distribution types
- classify_pearson: main classifier based on skewness and kurtosis
- pearson_moments: helper for computing β₁, β₂, D
- get_distribution: mapping Pearson types to SciPy distributions
"""

import math
from enum import StrEnum

from scipy.stats import beta, betaprime, gamma, invgamma, norm, t

from scipy_analytics.stats.distributions.scipy_wrappers import SciPyDistribution


class PearsonType(StrEnum):
    """
    Enumeration of Pearson distribution types.

    Members
    -------
    NORMAL
        Symmetric distribution with skew = 0 and kurtosis = 3.
    I
        Pearson Type I (Beta distribution on finite interval).
    II
        Pearson Type II (Symmetric Beta distribution).
    III
        Pearson Type III (Gamma distribution).
    IV
        Pearson Type IV (skewed distribution with heavy tails).
    V
        Pearson Type V (Inverse Gamma distribution).
    VI
        Pearson Type VI (Beta prime distribution).
    VII
        Pearson Type VII (Student's t distribution family).
    UNKNOWN
        Returned when classification is not possible.

    Notes
    -----
    - Types I and II correspond to Beta distributions.
    - Type VII corresponds to Student’s t and related heavy‑tailed families.
    - Type IV is the most general and least commonly used.
    """

    NORMAL = "Normal"
    I = "Pearson I"
    II = "Pearson II"
    III = "Pearson III"
    IV = "Pearson IV"
    V = "Pearson V"
    VI = "Pearson VI"
    VII = "Pearson VII"
    UNKNOWN = "Unknown"


PEARSON_MAP = {
    PearsonType.I: beta,
    PearsonType.II: beta,
    PearsonType.III: gamma,
    PearsonType.VII: t,
    PearsonType.V: invgamma,
    PearsonType.VI: betaprime,
    PearsonType.NORMAL: norm,
}


def pearson_moments(skew: float, kurt: float) -> tuple[float, float, float]:
    """
    Compute Pearson moment parameters β₁, β₂, and D.

    Parameters
    ----------
    skew : float
        Sample skewness.
    kurt : float
        Sample kurtosis (Fisher definition).

    Returns
    -------
    tuple
        (beta1, beta2, D) where:
        - beta1 = skew²
        - beta2 = kurt
        - D = beta2 − beta1 − 1

    Notes
    -----
    D determines the Pearson type:
    - D < 0 → Beta family (Types I, II)
    - D = 0 → special cases
    - D > 0 → Types III–VII
    """
    beta1 = skew**2
    beta2 = kurt
    D = beta2 - beta1 - 1
    return beta1, beta2, D


def classify_special(skew: float, kurt: float) -> PearsonType | None:
    """
    Handle special Pearson cases.

    Returns
    -------
    PearsonType or None
        NORMAL if skew=0 and kurt=3,
        VII if skew=0 and kurt>3,
        otherwise None.
    """
    if skew == 0 and kurt == 3:
        return PearsonType.NORMAL
    if skew == 0 and kurt > 3:
        return PearsonType.VII
    return None


def classify_beta_family(skew: float, D: float) -> PearsonType | None:
    """
    Classify distributions belonging to the Beta family (Types I and II).

    Parameters
    ----------
    skew : float
        Sample skewness.
    D : float
        Pearson D parameter.

    Returns
    -------
    PearsonType or None
        PearsonType.I or PearsonType.II, otherwise None.
    """
    if D < 0:
        return PearsonType.II if skew == 0 else PearsonType.I
    return None


def classify_remaining(skew: float, kurt: float, D: float) -> PearsonType | None:
    """
    Classify remaining Pearson types (III, IV, V, VI).

    Returns
    -------
    PearsonType or None
        One of PearsonType.III, IV, V, VI, or None.
    """
    if D > 0 and skew != 0:
        return PearsonType.IV
    if skew > 0 and kurt > 3:
        return PearsonType.III
    if skew < 0 and kurt > 3:
        return PearsonType.V
    if skew != 0 and kurt < 3:
        return PearsonType.VI
    return None


def classify_pearson(skew: float, kurt: float) -> PearsonType:
    """
    Classify a distribution according to the Pearson System.

    Parameters
    ----------
    skew : float
        Sample skewness.
    kurt : float
        Sample kurtosis.

    Returns
    -------
    PearsonType
        The Pearson type corresponding to the given skewness/kurtosis.

    Notes
    -----
    - Returns UNKNOWN if classification is not possible.
    - NaN skew/kurt also produce UNKNOWN.
    """
    if math.isnan(skew) or math.isnan(kurt):
        return PearsonType.UNKNOWN

    beta1 = skew**2
    beta2 = kurt
    D = beta2 - beta1 - 1

    if t := classify_special(skew, kurt):
        return t

    if t := classify_beta_family(skew, D):
        return t

    if t := classify_remaining(skew, kurt, D):
        return t

    return PearsonType.UNKNOWN


def get_distribution(dist_type: PearsonType, params: tuple) -> SciPyDistribution:
    """
    Map a Pearson type to the corresponding SciPy distribution.

    Parameters
    ----------
    dist_type : PearsonType
        The Pearson type (e.g., PearsonType.I, PearsonType.III).
    params : tuple
        Distribution parameters (shape, location, scale).

    Returns
    -------
    SciPyDistribution
        A unified distribution wrapper providing pdf, cdf, rvs, etc.

    Raises
    ------
    ValueError
        If the Pearson type is not implemented.

    Notes
    -----
    - Types I and II → Beta distribution
    - Type III → Gamma distribution
    - Type V → Inverse Gamma
    - Type VI → Beta prime
    - Type VII → Student's t
    - NORMAL → Gaussian
    """
    match dist_type:
        case t if t in PEARSON_MAP:
            dist = PEARSON_MAP[t]
            return SciPyDistribution(dist, *params)

        case _:
            raise ValueError(f"Pearson type '{dist_type}' not implemented")
