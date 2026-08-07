import matplotlib
matplotlib.use("Agg")  # no GUI

import numpy as np

import matplotlib.pyplot as plt
import warnings

plt.show = lambda *args, **kwargs: None

from scipy_analytics.stats.correlation.pearson.region_plot import (
    plot_pearson_region,
    plot_pearson_distribution,
    plot_pearson_classification_map,
    plot_skew_kurtosis_plane,
    plot_distribution_fit,
)
from scipy_analytics.stats.correlation.pearson.classification import PearsonType


def test_plot_pearson_region():
    plot_pearson_region(0.5)


def test_plot_pearson_distribution():
    plot_pearson_distribution(PearsonType.NORMAL, (0, 1))


def test_plot_pearson_classification_map():
    plot_pearson_classification_map()


def test_plot_skew_kurtosis_plane():
    plot_skew_kurtosis_plane(0.5, 3.2)


def test_plot_distribution_fit():
    data = np.random.normal(0, 1, 500)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plot_distribution_fit(data)