"""
Kernel Density Estimation (KDE) utilities.

The `density` subpackage provides:

- KDE class (wrapper around scikit‑learn KernelDensity)
- Silverman bandwidth selection
- Cross‑validated bandwidth selection (cv_bandwidth)
- Grid evaluation utilities for plotting

Modules
-------
kde
    Unified 1D KDE interface.
bandwidth
    Cross‑validated bandwidth selection.

Goals
-----
- Unified KDE API
- NumPy‑friendly interface
- Plug‑and‑play usage in statistical workflows
- Integration with plotting modules
"""
