import numpy as np


def sample_moments(data):
    """
    Υπολογίζει τις βασικές ροπές ενός δείγματος:
    μέση τιμή, διασπορά, ασυμμετρία και κύρτωση.

    Η μέθοδος αυτή αντιστοιχεί στη θεωρία των ροπών
    που χρησιμοποιείται για την εκτίμηση παραμέτρων
    των κατανομών κατά Pearson (method of moments).

    Parameters
    ----------
    data : array_like
        Το δείγμα δεδομένων.

    Returns
    -------
    tuple
        (mean, variance, skewness, kurtosis)
    """
    mean = np.mean(data)
    var = np.var(data)
    skew = ((data - mean)**3).mean() / var**1.5
    kurt = ((data - mean)**4).mean() / var**2
    return mean, var, skew, kurt