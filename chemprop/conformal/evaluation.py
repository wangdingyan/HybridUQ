import numpy as np


def reliability(y: np.ndarray,
                p_low: np.ndarray,
                p_high: np.ndarray):
    return ((p_low < y) * (y < p_high)).mean()


def interval(p_low: np.ndarray,
             p_high: np.ndarray):
    return np.abs(p_high - p_low)
