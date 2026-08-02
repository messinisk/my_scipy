import matplotlib.pyplot as plt


def plot_histogram(data, bins="auto", density=True):
    """
    Σχεδιάζει ιστόγραμμα των δεδομένων.

    Το ιστόγραμμα αποτελεί εμπειρική εκτίμηση της PDF και
    χρησιμοποιείται για σύγκριση της πραγματικής κατανομής
    με θεωρητικά μοντέλα Pearson ή άλλες συνεχείς κατανομές.

    Parameters
    ----------
    data : array_like
        Τα δεδομένα προς απεικόνιση.
    bins : int or str, optional
        Πλήθος ή στρατηγική επιλογής κάδων.
    density : bool, optional
        Αν True, το ιστόγραμμα κανονικοποιείται ώστε να
        προσεγγίζει PDF.

    Returns
    -------
    None
    """
    plt.hist(data, bins=bins, density=density, histtype="stepfilled", alpha=0.3)
    plt.title("Histogram")
    plt.xlabel("Value")
    plt.ylabel("Density" if density else "Count")
    plt.show()
