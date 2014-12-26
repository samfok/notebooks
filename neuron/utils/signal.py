# utilities for generating and processing signals
import numpy as np


def make_poisson_spikes(rate, nspikes, rng=np.random):
    """Creates spikes with Poisson statistics

    Parameters
    ----------
    rate : float
    nspikes : int
    rng : numpy random number generator
    """
    if rate == 0:
        return np.array([])
    beta = 1./rate
    isi = rng.exponential(beta, nspikes)
    spike_times = np.cumsum(isi)
    if isinstance(spike_times, (float, int)):
        spike_times = np.array([spike_times])
    return spike_times


def make_uniform_spikes(rate, nspikes, rng=np.random):
    """Creates spikes with Poisson statistics

    Parameters
    ----------
    rate : float
    nspikes : int
    rng : numpy random number generator
        used to set the offset of the first spike
    """
    if rate == 0:
        return np.array([])
    isi = 1./rate
    offset = isi*rng.uniform()
    spike_times = isi*(np.arange(nspikes)+offset)
    if isinstance(spike_times, (float, int)):
        spike_times = np.array([spike_times])
    return spike_times

def filter_spikes(dt, duration, spike_times, tau, ret_time=True):
    """Filters spikes with a synapse (first order low-pass filter)
    
    Assumes bins small enough that rounding spike times doesn't matter

    Parameters
    ----------
    dt : float
        Time step of filter
    duration : float
        Length of time to consider. Spikes after this time are ignored
    spike_times : array-like (floats)
        Times of spikes
    tau : float
        Filter time constant
    ret_time : boolean (optional)
        Whether to also return the time array corresponding the the state array
    """
    nbins = int(np.ceil(duration/dt))
    time = np.arange(nbins)*dt
    state = np.zeros(nbins)

    if spike_times.size == 0:
        return time, state

    assert isinstance(spike_times[0], (int, float))

    decay = np.exp(-dt/tau)
    spk_val = (1.-np.exp(-dt/tau))/dt
    for t in spike_times:
        bin_idx = int(round(t/dt))
        if bin_idx >= nbins:
            break
        state[bin_idx] += spk_val
    for idx in xrange(1, nbins):
        state[idx] += decay*state[idx-1]
    if ret_time:
        return time, state
    else:
        return state
