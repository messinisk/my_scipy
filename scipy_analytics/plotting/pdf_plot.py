import numpy as np
import matplotlib.pyplot as plt


def plot_pdf(dist, xmin, xmax, n=200)->None:
    """
    Σχεδιάζει την PDF μιας κατανομής στο διάστημα [xmin, xmax].

    Η μέθοδος αυτή αντιστοιχεί στη θεωρητική απεικόνιση της
    πυκνότητας πιθανότητας f(x), χρήσιμη για ανάλυση μορφής
    κατανομής και σύγκριση με θεωρητικά μοντέλα Pearson.

    Parameters
    ----------
    dist : Distribution
        Η κατανομή προς σχεδίαση.
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
    x = np.linspace(xmin, xmax, n)
    y = dist.pdf(x)
    plt.plot(x, y)
    plt.title("PDF")
    plt.show()