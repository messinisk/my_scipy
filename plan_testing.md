# Το δημοσιο  πλανο  testing

Επειδή το scipy θεωρειτε ενα αξιοπιστο πακετο θα  προσπαθησω  να  εγκαταστησω κωδικα ως  πακετω απευθειας  απο το github στο  google colab .  
1) Ετσι  μπορουμε να  δουμε  την εξοδο.
2) μπορουμε να  εξετασουμε  συμβατότητα σε ενα gui.
3) Αν τελικα  κωδικας  ειναι  επαρκης και επαναχρησιμοποιησιμος.


 version  0.0.1 => 0.1.1

Για  να  ειναι  ενα  φακελος  πακετω πρεπει  να  συνοδεβετε  setup.py ή pyproject.toml

## τροπος εγκαταστασης
### google colab :
 - !pip install git+https://github.com/messinisk/scipy_analytics.git
### Venv this is  wsl ubuntu
- pip install git+https://github.com/messinisk/scipy_analytics.git


## Το  πακετω εφανιζετε ως  
- pip show scipy-analytics
Name: scipy-analytics
Version: 0.1.1
Summary: Extensions for SciPy: Pearson distributions, Monte Carlo, fitting, plotting
Home-page:
Author: messinisk
Author-email: messinisk <messiniskostas0@gmail.com>
License: MIT
Location: /home/dev/test/lib/python3.14/site-packages
Requires: matplotlib, numpy, scipy
Required-by: