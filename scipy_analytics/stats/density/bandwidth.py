import numpy as np
from sklearn.model_selection import GridSearchCV

from scipy_analytics.stats.information.mutual_information import KernelDensity


def cv_bandwidth(data, bandwidths=None):
    if bandwidths is None:
        bandwidths = np.logspace(-2, 1, 20)

    grid = GridSearchCV(
        KernelDensity(kernel="gaussian"), {"bandwidth": bandwidths}, cv=5
    )
    grid.fit(data.reshape(-1, 1))
    return grid.best_params_["bandwidth"]
