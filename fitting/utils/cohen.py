import numpy as np


def cohen_w(tab):
    p1 = tab / np.sum(tab)
    p0 = np.dot(np.sum(p1, axis=1)[None].T, np.sum(p1, axis=0)[None])
    return np.sqrt(np.sum(np.power(p1 - p0, 2) / p0))
