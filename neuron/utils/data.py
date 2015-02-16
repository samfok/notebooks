# utility functions for managing data
import numpy as np
import os


def scalar_to_array(u):
    """If u is a scalar, converts it to a numpy array"""
    if isinstance(u, (int, float)):
        u = np.array([u])
    return u


def checkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)
