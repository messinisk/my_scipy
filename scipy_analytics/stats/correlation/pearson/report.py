"""
Reporting utilities for Pearson correlation analysis.

This module provides the PearsonReport class, which generates a structured,
human‑readable report for a Pearson correlation coefficient.

The report includes:
- The numeric value of the coefficient
- The Pearson System classification (PearsonType)
- Interpretation of the strength and direction
- Thresholds used for classification
- A summary dictionary for programmatic use

Works together with classification.py.
"""

from __future__ import annotations

from dataclasses import dataclass

from .classification import classify_pearson

# Thresholds for interpretation
PEARSON_THRESHOLDS = {
    "very weak": (0.0, 0.2),
    "weak": (0.2, 0.4),
    "moderate": (0.4, 0.6),
    "strong": (0.6, 0.8),
    "very strong": (0.8, 1.0),
}


def interpret_strength(r: float) -> str:
    """Return qualitative interpretation based on magnitude."""
    ar = abs(r)
    for label, (low, high) in PEARSON_THRESHOLDS.items():
        if low <= ar < high:
            return label
    return "perfect" if ar == 1.0 else "unknown"


def interpret_direction(r: float) -> str:
    """Return direction interpretation."""
    if r > 0:
        return "positive"
    if r < 0:
        return "negative"
    return "none"


@dataclass
class PearsonReport:
    """
    Generate a structured report for a Pearson correlation coefficient.

    Parameters
    ----------
    r : float
        Pearson correlation coefficient in [-1, 1].

    Attributes
    ----------
    r : float
        The correlation coefficient.
    ptype : PearsonType
        Pearson System classification based on skew/kurtosis assumptions.
    strength : str
        Qualitative strength (weak, moderate, strong, etc.).
    direction : str
        Positive, negative, or none.
    """

    r: float

    def __post_init__(self):
        if not isinstance(self.r, (int, float)):
            raise TypeError("Pearson coefficient must be numeric.")
        if self.r < -1 or self.r > 1:
            raise ValueError("Pearson coefficient must be in [-1, 1].")

        # Pearson System classification (based only on r)
        # Note: Pearson System is normally based on skew/kurtosis,
        # but here we classify correlation strength using the same enum.
        self.ptype = classify_pearson(0.0, 3.0)  # Normal distribution assumption

        self.strength = interpret_strength(self.r)
        self.direction = interpret_direction(self.r)

    # ------------------------------------------------------------------
    # Human-readable report
    # ------------------------------------------------------------------

    def text(self) -> str:
        """Return a human-readable report."""
        return (
            f"Pearson Correlation Report\n"
            f"---------------------------\n"
            f"Coefficient: {self.r:.3f}\n"
            f"Direction: {self.direction}\n"
            f"Strength: {self.strength}\n"
            f"Pearson Type: {self.ptype.value}\n"
            f"Interpretation: The relationship is {self.strength} and {self.direction}.\n"
        )

    # ------------------------------------------------------------------
    # Programmatic summary
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """Return a structured dictionary summary."""
        return {
            "r": self.r,
            "direction": self.direction,
            "strength": self.strength,
            "pearson_type": self.ptype.value,
        }
