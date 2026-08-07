"""
scipy-analytics: A unified statistical analysis toolkit.

This package provides:

- descriptive statistics
- distribution fitting
- density estimation (KDE, bandwidth selection)
- correlation and information metrics
- hypothesis testing
- quasi-Monte-Carlo sampling
- automated statistical reporting

The goal is to offer a clean, NumPy/SciPy-friendly API with
consistent return types, strong typing, and modular design.
"""

from __future__ import annotations

from . import stats

__version__ = "0.2.3"

__all__ = [
    "stats",
]
