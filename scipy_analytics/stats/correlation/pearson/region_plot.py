"""
Advanced visualization utilities for Pearson correlation and Pearson System.

This module provides:
- plot_pearson_region(r)
- plot_pearson_distribution(dist_type, params)
- plot_pearson_classification_map()
- plot_skew_kurtosis_plane(skew, kurt)
- plot_distribution_fit(data)

All plots use Matplotlib and are safe for notebooks, scripts, and reports.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kurtosis, skew

from scipy_analytics.stats.distributions.scipy_wrappers import SciPyDistribution

from .classification import (
    PEARSON_MAP,
    PearsonType,
    classify_pearson,
)

# ---------------------------------------------------------------------------
# 1) Pearson Region Plot (already implemented)
# ---------------------------------------------------------------------------


def plot_pearson_region(r: float) -> None:
    """Plot Pearson correlation coefficient on [-1, 1]."""
    if not isinstance(r, (int, float)):
        raise TypeError("Pearson coefficient must be numeric.")
    if r < -1 or r > 1:
        raise ValueError("Pearson coefficient must be in [-1, 1].")

    thresholds = [-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1]
    colors = [
        "#8b0000",
        "#b22222",
        "#cd5c5c",
        "#f08080",
        "#ffe4e1",
        "#e0e0e0",
        "#e0ffe1",
        "#90ee90",
        "#32cd32",
        "#228b22",
        "#006400",
    ]

    _, ax = plt.subplots(figsize=(10, 2))  # notype
    for i in range(len(thresholds) - 1):
        ax.axvspan(thresholds[i], thresholds[i + 1], color=colors[i], alpha=0.6)

    ax.axvline(r, color="black", linewidth=3)
    ax.text(
        r, 0.5, f"{r:.3f}", ha="center", va="center", fontsize=12, fontweight="bold"
    )

    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("Pearson Correlation Coefficient")
    ax.set_title("Pearson Correlation Region Plot")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 2) Pearson Distribution Plot
# ---------------------------------------------------------------------------


def plot_pearson_distribution(dist_type: PearsonType, params: tuple) -> None:
    """
    Plot the PDF of the distribution corresponding to a Pearson Type.

    Parameters
    ----------
    dist_type : PearsonType
        The Pearson type (I, II, III, IV, V, VI, VII, NORMAL).
    params : tuple
        Distribution parameters (shape, loc, scale).

    Notes
    -----
    - Uses SciPyDistribution wrapper for unified API.
    """
    if dist_type not in PEARSON_MAP:
        raise ValueError(f"Pearson type {dist_type} not supported.")

    dist = SciPyDistribution(PEARSON_MAP[dist_type], *params)

    x = np.linspace(dist.ppf(0.001), dist.ppf(0.999), 500)
    y = dist.pdf(x)

    plt.figure(figsize=(8, 4))
    plt.plot(x, y, label=f"{dist_type.value} PDF")
    plt.title(f"Pearson Distribution: {dist_type.value}")
    plt.xlabel("x")
    plt.ylabel("pdf")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 3) Pearson Classification Map (skew–kurtosis regions)
# ---------------------------------------------------------------------------


def plot_pearson_classification_map() -> None:
    """
    Plot the Pearson classification map in the skew–kurtosis plane.

    Shows regions corresponding to Pearson Types I–VII.
    """
    skew_vals = np.linspace(-3, 3, 400)
    kurt_vals = np.linspace(1, 15, 400)

    S, K = np.meshgrid(skew_vals, kurt_vals)
    D = K - S**2 - 1

    plt.figure(figsize=(8, 6))
    plt.contourf(S, K, D, levels=20, cmap="coolwarm")
    plt.colorbar(label="D = kurt - skew² - 1")

    plt.title("Pearson Classification Map (Skew–Kurtosis Plane)")
    plt.xlabel("Skewness")
    plt.ylabel("Kurtosis")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 4) Plot point on Pearson plane
# ---------------------------------------------------------------------------


def plot_skew_kurtosis_plane(skew_val: float, kurt_val: float) -> None:
    """
    Plot a single point (skew, kurtosis) on the Pearson plane.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(
        skew_val, kurt_val, s=200, c="red", label=f"({skew_val:.2f}, {kurt_val:.2f})"
    )
    plt.title("Skew–Kurtosis Pearson Plane")
    plt.xlabel("Skewness")
    plt.ylabel("Kurtosis")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 5) Distribution Fit Plot
# ---------------------------------------------------------------------------


def plot_distribution_fit(data: np.ndarray) -> None:
    """
    Fit a Pearson distribution to data and plot the empirical histogram
    together with the fitted PDF.
    """
    # Cast to Python floats for Pylance/mypy
    s = float(skew(data))
    k = float(kurtosis(data, fisher=True))

    ptype = classify_pearson(s, k)

    if ptype not in PEARSON_MAP:
        raise ValueError("Cannot fit Pearson distribution for UNKNOWN type.")

    # Use SciPy distribution directly for fitting
    scipy_dist = PEARSON_MAP[ptype]
    params = scipy_dist.fit(data)

    # Wrap fitted distribution
    fitted = SciPyDistribution(scipy_dist, *params)

    x = np.linspace(min(data), max(data), 500)
    y = fitted.pdf(x)

    plt.figure(figsize=(8, 5))
    plt.hist(data, bins=30, density=True, alpha=0.5, label="Empirical")
    plt.plot(x, y, label=f"Fitted {ptype.value} PDF", linewidth=2)
    plt.title(f"Pearson Fit: {ptype.value}")
    plt.xlabel("x")
    plt.ylabel("density")
    plt.legend()
    plt.tight_layout()
    plt.show()
