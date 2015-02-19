# utility functions for managing data
import numpy as np
import os


def scalar_to_array(x):
    """If x is a scalar, converts it to a numpy array"""
    if isinstance(x, (int, float)):
        x = np.array([x])
    return x


def checkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)
