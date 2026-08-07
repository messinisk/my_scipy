"""
Utility tools for data preprocessing and transformation.

This module provides small helper functions used across the
`scipy_analytics.stats` package. These utilities focus on:

- safe conversion of inputs to NumPy arrays
- validation of numeric data
- reshaping helpers for 1D/2D statistical routines
- detection of degenerate inputs
- small convenience wrappers used by KDE, MI, fitting, and descriptive stats

The module intentionally contains lightweight, dependency‑free utilities.
It acts as a shared foundation for higher‑level statistical components.

Notes
-----
- This file previously contained placeholder content and has now been
  replaced with a documented skeleton for future expansion.
- No browser metadata or unrelated content should exist in this module.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def to_numpy(data: Any) -> np.ndarray:
    """
    Convert input to a NumPy array.

    Parameters
    ----------
    data : Any
        Input data (list, tuple, NumPy array, pandas Series, etc.).

    Returns
    -------
    np.ndarray
        Converted array.

    Notes
    -----
    - Scalars are converted to 1‑element arrays.
    - Raises ValueError for empty sequences.
    """
    arr = np.asarray(data)

    if arr.size == 0:
        raise ValueError("Empty input data.")

    return arr


def ensure_1d(data: np.ndarray) -> np.ndarray:
    """
    Ensure that data is 1D.

    Parameters
    ----------
    data : np.ndarray

    Returns
    -------
    np.ndarray
        1D array.

    Notes
    -----
    - 2D arrays with shape (n, 1) are flattened.
    - Higher‑dimensional arrays raise ValueError.
    """
    if data.ndim == 1:
        return data

    if data.ndim == 2 and data.shape[1] == 1:
        return data.ravel()

    raise ValueError("Expected 1D data.")


def ensure_2d_column(data: np.ndarray) -> np.ndarray:
    """
    Ensure that data is shaped as (n, 1).

    Parameters
    ----------
    data : np.ndarray

    Returns
    -------
    np.ndarray
        2D column vector.

    Notes
    -----
    - 1D arrays are reshaped to (n, 1).
    - 2D arrays are returned unchanged if they already have shape (n, 1).
    """
    if data.ndim == 1:
        return data.reshape(-1, 1)

    if data.ndim == 2 and data.shape[1] == 1:
        return data

    raise ValueError("Expected data shaped as (n, 1).")


def is_constant(data: np.ndarray) -> bool:
    """
    Check whether all values in the array are identical.

    Parameters
    ----------
    data : np.ndarray

    Returns
    -------
    bool
        True if all values are equal, False otherwise.

    Notes
    -----
    - Used by mutual information and correlation modules.
    """
    return bool(np.all(data == data[0]))


def normalize_data(data: np.ndarray) -> np.ndarray:
    """
    Normalize data to zero mean and unit variance.

    Parameters
    ----------
    data : np.ndarray
        Input numeric array.

    Returns
    -------
    np.ndarray
        Normalized array: (data - mean) / std.

    Notes
    -----
    - If std == 0 (constant data), returns zeros.
    - Used by various statistical preprocessing steps.
    """
    arr = np.asarray(data, dtype=float)

    mean = arr.mean()
    std = arr.std()

    if std == 0:
        return np.zeros_like(arr)

    return (arr - mean) / std
