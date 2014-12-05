# utilities for generating and processing signals
import numpy as np


def make_poisson_spikes(rate, nspikes, rng=np.random):
    if rate == 0:
        return np.array([])
    beta = 1./rate
    isi = rng.exponential(beta, nspikes)
    spike_times = np.cumsum(isi)
    if isinstance(spike_times, (float, int)):
        spike_times = np.array([spike_times])
    return spike_times


def filter_spikes(dt, duration, spike_times, tau):
    # assume bins small enough that rounding spike times doesn't matter
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
    return time, state
