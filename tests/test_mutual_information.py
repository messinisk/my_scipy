from scipy_analytics.stats.information.mutual_information import mutual_information_continuous, mutual_information_discrete, kde_density

import numpy as np
import pytest

def test_mutual_information_independent():
    np.random.seed(0)
    x = np.random.randn(1000)
    y = np.random.randn(1000)

    mi = mutual_information_continuous(x, y)

    assert mi == pytest.approx(0.0, abs=0.1)

def test_mutual_information_linear():
    np.random.seed(0)
    x = np.random.randn(1000)
    y = 4*x + 1

    mi = mutual_information_continuous(x, y)

    assert mi > 0.5

def test_mutual_information_nonlinear():
    np.random.seed(0)
    x = np.random.uniform(-3, 3, 1000)
    y = np.sin(x)

    mi = mutual_information_continuous(x, y)

    assert mi > 1.0

def mutual_information_continuous(x, y, bw=0.3, samples=2000):
    # Degenerate input → MI undefined
    if np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan")

    xy = np.vstack([x, y]).T
    kde_xy = kde_density(xy, bw)
    kde_x = kde_density(x.reshape(-1, 1), bw)
    kde_y = kde_density(y.reshape(-1, 1), bw)

    pts = xy[np.random.choice(len(x), samples)]
    log_pxy = kde_xy.score_samples(pts)
    log_px = kde_x.score_samples(pts[:, [0]])
    log_py = kde_y.score_samples(pts[:, [1]])

    return np.mean(log_pxy - log_px - log_py)
