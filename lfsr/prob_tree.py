import numpy as np
from lfsr import LFSR

def AsBBit(x, B):
    """Converts a real number to a B-bit representation

    Rounds probabilistically
    """
    assert x < 1, "it's a probability"
    M = 2**B

    scaled_x = x*M
    rem = scaled_x - np.floor(scaled_x)

    r = np.random.rand()
    if (r < rem):
        return np.floor(scaled_x)
    else:
        return np.floor(scaled_x) + 1

def extract_distribution(w)
    """Extract distribution from ptree in binary heap format"""
    w_apx = np.zeros(h.shape)
    FlattenPtreeRecur(h, w_apx, 1, 1)
    return w_apx

def FlattenPtreeRecur(h, w_apx, i, x):
    if (i < len(h)):
        x_l = x * h[i]
        x_r = x * (1 - h[i])
        FlattenPtreeRecur(h, w_apx, 2*i, x_l)
        FlattenPtreeRecur(h, w_apx, 2*i + 1, x_r)
    else:
        assert(i < 2*len(h) and i >= len(h))
        w_apx[i - len(h)] = x 

class ProbTree(object):
    """Represents a weight distribution as a probability tree

    Attributes
    ----------
    w : weight distribution to be approximated
    tree : probability tree representation of w in binary heap format
    w_apx : weight distribution implemented by tree
    """

    def __init__(self, nbits, lfsr_seed=0b1, w_dist=None):
        assert nbits >1, "must use more than 1 bit"
        self.nbits = nbits
        self.lfsr = LFSR(nbits, lfsr_seed)

        if w_dist is not None:
            self.build_tree(w_dist)
        else:
            self.w = None
            self.tree = None
            self.w_apx = None

    def build_tree(self, w):
        """builds the probability tree from a weight distribution
        """
        self.w = w
        w_abs = np.abs(w)
        self.tree = np.zeros(w.shape)
        self._build_node(w_abs, 1)
        self.w_apx = extract_distribution(self.tree)

    def _build_node(self, w, i):
        if (i < len(w)):
            sum_left = self._build_node(w, 2*i)
            sum_right = self._build_node(w, 2*i + 1)

            p_left = sum_left/(sum_right + sum_left)
            self.tree[i] = AsBBit(p_left, self.nbits) / 2**self.nbits

            return sum_left + sum_right
        else:
            return w[i-len(w)]

    def sample_tree(self):
        i = 1
        len_w = self.tree.shape[0]

        while i<len_w:
            r = self.lfsr.get_next_state()
            if r<self.tree[i]:
                i = i*2
            else:
                i = i*2 + 1
        return i
