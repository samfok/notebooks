# defines data types used in accumulator experiments
import numpy as np

class SpikeTrain(object):
    """Container for trains of spikes
    """
    def __init__(self, times, weights):
        if not hasattr(weights, "__iter__"):
            self.times = [times]
        else:
            self.times = times
        if not hasattr(weights, "__iter__"):
            self.weights = [weights for n in range(len(self.times))]
        else:
            self.weights = weights

    def __getitem__(self, index):
        return SpikeTrain(
            np.array(self.times)[index], np.array(self.weights)[index])

class AccumulatorState(object):
    """Container for accumulator state
    """
    def __init__(self, time, state):
        self.time = time
        self.state = state

    def __getitem__(self, index):
        return AccumulatorState(self.time[index], self.state[index])

def set_list_var(x, N):
    """Expands a variable into a list if it is not already a list
    """
    if not hasattr(x, "__iter__"):
        return [x for n in range(N)]
    else:
        return x
