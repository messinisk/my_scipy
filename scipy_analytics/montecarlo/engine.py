

class MonteCarlo:
    """
    Κινητήρας Monte Carlo για προσομοιώσεις με βάση
    οποιαδήποτε συνεχής κατανομή.

    Η κλάση αυτή υλοποιεί την θεωρητική διαδικασία
    Monte Carlo: δειγματοληψία από μια κατανομή και
    υπολογισμό στατιστικών ή λειτουργικών μεγεθών.

    Parameters
    ----------
    distribution : Distribution
        Αντικείμενο κατανομής που ακολουθεί το ενιαίο API.

    Methods
    -------
    simulate(n)
        Παράγει n τυχαίες τιμές από την κατανομή.

    experiment(n, func)
        Εκτελεί πείραμα Monte Carlo εφαρμόζοντας μια
        συνάρτηση func στα παραγόμενα δείγματα.
    """
    def __init__(self, distribution):
        self.dist = distribution

    def simulate(self, n):
        """
        simulate(n)
        Παράγει n τυχαίες τιμές από την κατανομή.
        """
        return self.dist.rvs(size=n)

    def experiment(self, n, func):
        """
        Εκτελεί πείραμα Monte Carlo εφαρμόζοντας μια
        συνάρτηση func στα παραγόμενα δείγματα.
        """
        samples = self.simulate(n)
        return func(samples)