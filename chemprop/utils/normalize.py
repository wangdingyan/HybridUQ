import numpy as np


def min_max(x):
    return (x-np.min(x)) / (np.max(x)-np.min(x)) + 1e-6