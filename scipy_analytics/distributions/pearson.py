from enum import Enum
import numpy as np
from scipy.stats import beta, gamma, t, invgamma, betaprime, norm
from .scipy_wrappers import SciPyDistribution

class PearsonType(Enum):
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
        PearsonType.NORMAL: norm,}


def classify_pearson(skew, kurt):
    """
    Ταξινομεί μια κατανομή στον κατάλληλο τύπο Pearson
    με βάση την ασυμμετρία (skewness) και την κύρτωση (kurtosis).

    Η μέθοδος υλοποιεί τον θεωρητικό κανόνα ταξινόμησης
    του Karl Pearson, ο οποίος βασίζεται στις ροπές:
    β1 = skew^2 και β2 = kurtosis.

    Parameters
    ----------
    skew : float
        Η ασυμμετρία του δείγματος.
    kurt : float
        Η κύρτωση του δείγματος.

    Returns
    -------
    str
        Το όνομα του τύπου Pearson (I, III, IV, VII, κ.λπ.).
    """

    beta1 = skew**2
    beta2 = kurt
    D = beta2 - beta1 - 1

    match (skew, kurt, D):

        # Κανονική κατανομή
        case (0, 3, _):
            return PearsonType.NORMAL

        # Pearson VII (Student-t γενίκευση)
        case (0, kurt, _) if kurt > 3:
            return PearsonType.VII

         # Pearson I / II (Beta family)
        case (_, _, D) if D < 0:
            if skew == 0:
                return PearsonType.II
            return PearsonType.I

        # Pearson IV (complex roots)
        case (skew, _, D) if D > 0 and skew != 0:
            return PearsonType.IV

        # Pearson III (Gamma)
        case (skew, kurt, _) if skew > 0 and kurt > 3:
            return PearsonType.III
        
        # Pearson V (Inverse Gamma)
        case (skew, kurt, _) if skew < 0 and kurt > 3:
            return PearsonType.V

        # Pearson VI (Beta Prime)
        case (skew, kurt, _) if skew != 0 and kurt < 3:
            return PearsonType.VI

        # Άγνωστο / μη ταξινομήσιμο
        case _:
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