from scipy.stats import rv_continuous
from .base import Distribution


class SciPyDistribution(Distribution):
    """
    Περιτύλιξη (wrapper) γύρω από τις κατανομές του SciPy.

    Η κλάση αυτή επιτρέπει τη χρήση των κατανομών του SciPy
    μέσα από ένα ενιαίο και καθαρό API, συμβατό με το θεωρητικό
    μοντέλο των κατανομών κατά Pearson. Μετατρέπει τις global
    συναρτήσεις του SciPy σε αντικείμενα με μεθόδους.

    Parameters
    ----------
    dist : scipy.stats.rv_continuous
        Η κατανομή του SciPy που θα περιτυλιχθεί.
    *params : float
        Οι παράμετροι της κατανομής (shape, loc, scale).

    Methods
    -------
    pdf(x)
        Υπολογίζει την PDF της κατανομής.

    cdf(x)
        Υπολογίζει την CDF της κατανομής.

    ppf(q)
        Υπολογίζει την PPF της κατανομής.

    rvs(size=1)
        Παράγει τυχαίες τιμές από την κατανομή.

    stats()
        Επιστρέφει τις ροπές της κατανομής (mvsk).
    """

    def __init__(self, dist: rv_continuous, *params):
        self.dist = dist
        self.params = params

    def pdf(self, x):
        """
        pdf(x)
        Υπολογίζει την PDF της κατανομής.
        """
        return self.dist.pdf(x, *self.params)

    def cdf(self, x):
        """
        cdf(x)
        Υπολογίζει την CDF της κατανομής.
        """
        return self.dist.cdf(x, *self.params)

    def ppf(self, q):
        """
        ppf(q)
        Υπολογίζει την PPF της κατανομής.
        """
        return self.dist.ppf(q, *self.params)

    def rvs(self, size=1):
        """
        rvs(size=1)
        Παράγει τυχαίες τιμές από την κατανομή.
        """
        return self.dist.rvs(*self.params, size=size)

    def stats(self):
        """
        stats()
        Επιστρέφει τις ροπές της κατανομής (mvsk).
        
        """
        return self.dist.stats(*self.params, moments='mvsk')
