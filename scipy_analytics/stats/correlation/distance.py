import numpy as np
from scipy.spatial.distance import pdist, squareform


def distance_matrix(x):
    return squareform(pdist(x.reshape(-1, 1)))


def double_center(D):
    row_mean = D.mean(axis=1, keepdims=True)
    col_mean = D.mean(axis=0, keepdims=True)
    total_mean = D.mean()
    return D - row_mean - col_mean + total_mean


def distance_covariance(x, y):
    A = double_center(distance_matrix(x))
    B = double_center(distance_matrix(y))
    return np.sqrt(np.mean(A * B))


def distance_variance(x):
    A = double_center(distance_matrix(x))
    return np.sqrt(np.mean(A * A))


def distance_correlation(x, y):

    dcov = distance_covariance(x, y)
    dvar_x = distance_variance(x)
    dvar_y = distance_variance(y)
    if dvar_x == 0 or dvar_y == 0:
        return float("nan")
    return dcov / np.sqrt(dvar_x * dvar_y)
