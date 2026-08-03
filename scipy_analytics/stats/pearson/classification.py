# pearson.py
from enum import StrEnum
import math

from scipy.stats import beta, betaprime, gamma, invgamma, norm, t

from scipy_analytics.distributions.scipy_wrappers import SciPyDistribution


class PearsonType(StrEnum):
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
    beta1 = skew ** 2
    beta2 = kurt
    D = beta2 - beta1 - 1
    return beta1, beta2, D

def classify_special(skew: float, kurt: float) -> str | None:
    if skew == 0 and kurt == 3:
        return PearsonType.NORMAL
    if skew == 0 and kurt > 3:
        return PearsonType.VII
    return None

def classify_beta_family(skew: float, D: float) -> str | None:
    if D < 0:
        return PearsonType.II if skew == 0 else PearsonType.I
    return None

def classify_remaining(skew: float, kurt: float, D: float) -> PearsonType | None:

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
    # Αν οι ροπές δεν είναι ορισμένες
    if math.isnan(skew) or math.isnan(kurt):
        return PearsonType.UNKNOWN

    beta1 = skew ** 2
    beta2 = kurt
    D = beta2 - beta1 - 1

    if (t := classify_special(skew, kurt)):
        return t # type: ignore

    if (t := classify_beta_family(skew, D)):
        return t # type: ignore

    if (t := classify_remaining(skew, kurt, D)):
        return t

    return PearsonType.UNKNOWN
 
def get_distribution(dist_type, params):
    """
    Επιστρέφει το αντικείμενο κατανομής που αντιστοιχεί
    στον δοθέντα τύπο Pearson.

    Η μέθοδος αυτή συνδέει τη θεωρητική ταξινόμηση Pearson
    με τις αντίστοιχες κατανομές του SciPy (Beta, Gamma, t κ.λπ.).

    Parameters
    ----------
    dist_type : str
        Ο τύπος Pearson (π.χ. 'Pearson I', 'Pearson III').
    params : tuple
        Οι παράμετροι της κατανομής.

    Returns
    -------
    SciPyDistribution
        Αντικείμενο κατανομής με ενιαίο API.
    """

    match dist_type:
        case t if t in PEARSON_MAP:
            dist = PEARSON_MAP[t]
            return SciPyDistribution(dist, *params)

        case _:
            raise ValueError(f"Pearson type '{dist_type}' not implemented")
