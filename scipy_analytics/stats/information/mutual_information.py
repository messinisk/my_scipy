import numpy as np
from sklearn.neighbors import KernelDensity


def kde_density(data: np.ndarray, bw: float = 0.3) -> KernelDensity:
    kde = KernelDensity(bandwidth=bw, kernel="gaussian")
    kde.fit(data)
    return kde


def mutual_information_discrete(x, y, bins=20):
    joint_hist, _, _ = np.histogram2d(x, y, bins=bins)
    joint_prob = joint_hist / joint_hist.sum()

    px = joint_prob.sum(axis=1)
    py = joint_prob.sum(axis=0)

    mi = 0.0
    for i in range(len(px)):
        for j in range(len(py)):
            if joint_prob[i, j] > 0:
                mi += joint_prob[i, j] * np.log(joint_prob[i, j] / (px[i] * py[j]))
    return mi


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
