import numpy as np
import matplotlib.pyplot as plt

def plot_cdf(dist, xmin, xmax, n=200)-> None:
    """
    Σχεδιάζει την αθροιστική συνάρτηση πιθανότητας (CDF)
    της κατανομής στο διάστημα [xmin, xmax].

    Η CDF αντιστοιχεί στη θεωρητική συνάρτηση F(x) = P(X ≤ x),
    η οποία αποτελεί βασικό εργαλείο στην ανάλυση κατανομών
    κατά Pearson και γενικότερα στη στατιστική συμπερασματολογία.

    Parameters
    ----------
    dist : Distribution
        Αντικείμενο κατανομής που υλοποιεί το ενιαίο API.
    xmin : float
        Κάτω όριο του άξονα x.
    xmax : float
        Άνω όριο του άξονα x.
    n : int, optional
        Πλήθος σημείων δειγματοληψίας.

    Returns
    -------
    None
    """
    # Δημιουργία σημείων στο διάστημα [xmin, xmax]
    x = np.linspace(xmin, xmax, n)

    # Υπολογισμός CDF για κάθε σημείο
    y = dist.cdf(x)

    # Σχεδίαση
    plt.plot(x, y, lw=2)
    plt.title("CDF")
    plt.xlabel("x")
    plt.ylabel("F(x)")
    plt.grid(True)
    plt.show()
