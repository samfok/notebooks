# utility functions for managing data
import os


def checkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)
