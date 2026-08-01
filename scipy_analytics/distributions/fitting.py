import numpy as np
from scipy.stats import rv_continuous


def fit_moments(data):
    """
    Εκτίμηση παραμέτρων κατανομής με τη μέθοδο των ροπών.

    Η μέθοδος των ροπών αποτελεί κλασική τεχνική στη θεωρία
    Pearson για την εκτίμηση παραμέτρων από δείγμα, χρησιμοποιώντας
    τις ροπές: μέση τιμή, διασπορά, ασυμμετρία και κύρτωση.

    Parameters
    ----------
    data : array_like
        Το δείγμα δεδομένων.

    Returns
    -------
    dict
        Λεξικό με τις εκτιμημένες ροπές και παραμέτρους.
    """
    data = np.asarray(data)

    mean = np.mean(data)
    var = np.var(data)
    skew = np.mean((data - mean)**3) / (var**1.5)
    kurt = np.mean((data - mean)**4) / (var**2)

    return {
        "mean": float(mean),
        "variance": float(var),
        "skewness": float(skew),
        "kurtosis": float(kurt)
    }




def fit_distribution(dist_class, data):
    """
    Εκτίμηση παραμέτρων κατανομής μέσω βελτιστοποίησης.

    Η μέθοδος αυτή αντιστοιχεί στη θεωρητική διαδικασία
    προσαρμογής κατανομής (distribution fitting), όπου
    οι παράμετροι εκτιμώνται με βάση την ελαχιστοποίηση
    κάποιας συνάρτησης κόστους (π.χ. MLE, least squares).

    Parameters
    ----------
    dist_class : type
        Κλάση κατανομής που θα προσαρμοστεί.
        Πρέπει να είναι συμβατή με rv_continuous του SciPy.
    data : array_like
        Το δείγμα δεδομένων.

    Returns
    -------
    Distribution
        Αντικείμενο κατανομής με εκτιμημένες παραμέτρους.
    """
    data = np.asarray(data)

    # SciPy fitting (MLE)
    params = dist_class.fit(data)

    # Επιστροφή ως SciPyDistribution αντικείμενο
    from .scipy_wrappers import SciPyDistribution
    return SciPyDistribution(dist_class, *params)
