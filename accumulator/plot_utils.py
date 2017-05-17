# defines some utilities for plotting
import numpy as np
from nengo.utils.matplotlib import rasterplot as nengo_raster

def rasterplot(spike_trains, ax=None):
    """Wrap nengo rasterplot for this notebook
    Inputs
    ------
    spike_trains: SpikeTrain or list of spike_trains
    
    Outputs
    -------
    handle of axis used to plot
    """
    if type(spike_trains) is not list:
        spike_trains = [spike_trains]
    N = len(spike_trains)
    T = []
    for n in range(N):
        T.append(len(spike_trains[n].times))
    spikes = np.zeros((np.sum(T), N+1))
    
    idx0 = 0
    for n in range(N):
        idx1 = idx0 + T[n]
        spikes[idx0:idx1, 0] = spike_trains[n].times
        spikes[idx0:idx1, n+1] = 1
        idx0 = idx1
    
    time = spikes[:,0]
    spikes = spikes[:,1:]
    
    return nengo_raster(time, spikes, ax=ax)
