import numpy as np


def mean_experiment(mc_engine, n):
    """
    Πείραμα Monte Carlo για εκτίμηση της μέσης τιμής.

    Η μέθοδος αυτή αντιστοιχεί στη θεωρητική διαδικασία
    Monte Carlo, όπου η μέση τιμή εκτιμάται μέσω δειγματοληψίας
    από την κατανομή. Χρήσιμη για έλεγχο σύγκλισης ροπών
    σε κατανομές Pearson.

    Parameters
    ----------
    mc_engine : MonteCarlo
        Ο κινητήρας Monte Carlo.
    n : int
        Πλήθος δειγμάτων.

    Returns
    -------
    float
        Εκτίμηση της μέσης τιμής.
    """
    samples = mc_engine.simulate(n)
    return float(np.mean(samples))


def variance_experiment(mc_engine, n):
    """
    Πείραμα Monte Carlo για εκτίμηση της διασποράς.

    Αντιστοιχεί στη θεωρητική 2η ροπή γύρω από τη μέση τιμή,
    η οποία αποτελεί βασικό στοιχείο ταξινόμησης Pearson.

    Parameters
    ----------
    mc_engine : MonteCarlo
        Ο κινητήρας Monte Carlo.
    n : int
        Πλήθος δειγμάτων.

    Returns
    -------
    float
        Εκτίμηση της διασποράς.
    """
    samples = mc_engine.simulate(n)
    return float(np.var(samples))


def skew_experiment(mc_engine, n):
    """
    Πείραμα Monte Carlo για εκτίμηση της ασυμμετρίας (skewness).

    Η ασυμμετρία αποτελεί θεμελιώδη παράμετρο στη θεωρία Pearson,
    καθώς καθορίζει τον τύπο κατανομής (β1 = skew^2).

    Parameters
    ----------
    mc_engine : MonteCarlo
        Ο κινητήρας Monte Carlo.
    n : int
        Πλήθος δειγμάτων.

    Returns
    -------
    float
        Εκτίμηση της ασυμμετρίας.
    """
    samples = mc_engine.simulate(n)
    mean = np.mean(samples)
    var = np.var(samples)
    skew = np.mean((samples - mean) ** 3) / (var**1.5)
    return float(skew)


def kurtosis_experiment(mc_engine, n):
    """
    Πείραμα Monte Carlo για εκτίμηση της κύρτωσης (kurtosis).

    Η κύρτωση αποτελεί βασική ροπή στη θεωρία Pearson και
    συμμετέχει στον κανόνα ταξινόμησης μέσω της β2.

    Parameters
    ----------
    mc_engine : MonteCarlo
        Ο κινητήρας Monte Carlo.
    n : int
        Πλήθος δειγμάτων.

    Returns
    -------
    float
        Εκτίμηση της κύρτωσης.
    """
    samples = mc_engine.simulate(n)
    mean = np.mean(samples)
    var = np.var(samples)
    kurt = np.mean((samples - mean) ** 4) / (var**2)
    return float(kurt)
