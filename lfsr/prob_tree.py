import numpy as np
from lfsr import LFSR

class ProbTree(object):
    def __init__(self, nbits, lfsr_seed=0b1):
        assert nbits >1, "must use more than 1 bit"
        self.nbits = nbits
        self.lfsr = LFSR(nbits, lfsr_seed)

    def build_tree(self):
        """builds the probability tree given a distribution"""



