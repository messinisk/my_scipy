"""
Pearson Analytics Package
=========================

Το πακέτο `pearson_analytics` παρέχει εργαλεία για ταξινόμηση,
ανάλυση και οπτικοποίηση Pearson correlation regions.

Κύριες λειτουργίες:

- classify_pearson:
    Υπολογίζει και ταξινομεί τον Pearson correlation coefficient
    σε προκαθορισμένους τύπους (βλ. PearsonType).

- PearsonType:
    Enum που περιγράφει τους διαθέσιμους τύπους Pearson correlation
    (π.χ. Strong Positive, Weak Negative κτλ).

- PearsonReport:
    Κλάση που δημιουργεί αναλυτική αναφορά για τον Pearson coefficient,
    περιλαμβάνοντας περιγραφή, thresholds και ερμηνεία.

- plot_pearson_region:
    Συνάρτηση οπτικοποίησης που εμφανίζει το Pearson region plot
    με βάση την ταξινόμηση και τις τιμές του coefficient.

Το `__all__` ορίζει τα δημόσια exports του πακέτου, ώστε να είναι
σαφές ποια API στοιχεία είναι διαθέσιμα στον χρήστη.
"""

from .classification import PearsonType, classify_pearson
from .region_plot import plot_pearson_region
from .report import PearsonReport

__all__ = [
    "PearsonReport",
    "PearsonType",
    "classify_pearson",
    "plot_pearson_region",
]
