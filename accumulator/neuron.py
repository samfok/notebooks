# defines a Neuron class and its variants
import numpy as np
import abc
from datatypes import SpikeTrain

class Neuron(object):
    """Abstract base neuron class
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, spike_rate, weight, T0=None):
        """Generates spike trains
        spike_rate: spike rate of neuron
        weight: decode weight of neuron
        T0: initial spike time
            default uniformly random 
        """
        assert type(weight) in [int, np.int, np.int64]
        assert spike_rate >= 0.
        self.spike_rate = spike_rate
        self.weight = weight

    @abc.abstractmethod
    def generate_spikes(self, T):
        """Generates spikes within a time period T
        """
        return

class RegularNeuron(Neuron):
    """ Subclass Neuron to implement a regular spiking neuron
    """
    def __init__(self, spike_rate, weight, T0=None):
        """Generates spike trains
        spike_rate: spike rate of neuron
        weight: decode weight of neuron
        T0: initial spike time
            default uniformly random 
        """
        super(RegularNeuron, self).__init__(spike_rate, weight)
        if spike_rate > 0.:
            self.period = 1./spike_rate
        if T0 == None:
            self.T0 = np.random.uniform(0, self.period)
        else:
            self.T0 = T0

    def generate_spikes(self, T):
        """Generates spikes within a time period T
        """
        if self.spike_rate > 0.:
            nspikes = int(T/self.period)
            times = self.period * np.arange(nspikes) + self.T0
            weights = self.weight * np.ones(nspikes, dtype=int)
        else:
            times = np.array([])
            weights = np.array([], dtype=int)
        spikes = SpikeTrain(times, weights)
        return spikes

class PoissonNeuron(Neuron):
    """ Subclass Neuron to implement a Poisson spiking neuron
    """
    def __init__(self, spike_rate, weight):
        super(PoissonNeuron, self).__init__(spike_rate, weight)
    
    def generate_spikes(self, T):
        """Generates spikes within a time period T
        """
        if self.spike_rate > 0.:
            nspikes = np.random.poisson(self.spike_rate * T)
            times = np.sort(np.random.uniform(low=0., high=T, size=nspikes))
            weights = self.weight * np.ones(nspikes, dtype=int)
        else:
            times = np.array([])
            weights = np.array([], dtype=int)
        spikes = SpikeTrain(times, weights)
        return spikes

