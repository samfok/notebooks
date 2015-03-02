import numpy as np


def rmse(x, y):
    err = x-y
    return np.sqrt(np.mean(err**2))
