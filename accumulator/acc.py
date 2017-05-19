# defins the accumulator
import numpy as np
from datatypes import SpikeTrain, AccumulatorState

class Accumulator(object):
    """Represents an accumulator
    Parameters:
        threshold: (int) threshold at which neuron will spike
    """
    def __init__(self, threshold):
        assert type(threshold) in [int, np.int, np.int64]
        self.threshold = threshold

    def accumulate(self, spikes_in):
        """Computes the output spike times of the accumulator
        Inputs:
            spikes_in:
        Outputs:
            spikes_out:
        """
        cur_state = 0
        state = [0]
        time = [0.]
        spike_times = []
        spike_weights = []

        for t, w in zip(spikes_in.times, spikes_in.weights):
            time.append(t)
            state.append(cur_state)
            cur_state += w
            time.append(t)
            state.append(cur_state)

            if cur_state >= self.threshold:
                spike_times.append(t)
                spike_weights.append(1)
                cur_state -= self.threshold
                time.append(t)
                state.append(cur_state)
            elif cur_state <= -self.threshold:
                spike_times.append(t)
                spike_weights.append(-1)
                cur_state += self.threshold
                time.append(t)
                state.append(cur_state)
        spikes_out = SpikeTrain(spike_times, spike_weights)
        state = AccumulatorState(time, state)
        return spikes_out, state
