# defines functions for running experiments
from datatypes import SpikeTrain, set_list_var
import matplotlib.pyplot as plt
import numpy as np
from neuron import RegularNeuron, PoissonNeuron
from plot_utils import rasterplot
from pool import Pool
from scipy.stats import gamma

def run_experiment(pool, T=None):
    """Runs an experiment
    """
    spks_in = pool.gen_nrn_spikes(T=T)
    merged_spks_in = pool.merge_spikes(spks_in)
    spks_out, acc_state = pool.gen_acc_spikes(merged_spks_in)
    return spks_in, acc_state, spks_out

def run_neuron_experiment(N, input_rates, weights, threshold,
    T=None, nspikes=None, neuron_model=RegularNeuron):
    """Run an experiment with a single neuron
    """
    spike_rates = set_list_var(input_rates, N)
    weights = set_list_var(weights, N)
    
    neurons = [neuron_model(spike_rate, weight) for 
               spike_rate, weight in
               zip(spike_rates, weights)]
    pool = Pool(neurons=neurons, threshold=threshold)
    spks_in = pool.gen_nrn_spikes(T=T, nspikes=nspikes)
    
    merged_spks_in = pool.merge_spikes(spks_in)
    spks_out, acc_state = pool.gen_acc_spikes(merged_spks_in)
    return spks_in, acc_state, spks_out

def limit_xlim(ax, minrange=0.002):
    """Keeps axis reasonable
    ax: axis handle
    minrange: minimum range
    """
    lim = ax.get_xlim()
    if lim[1] - lim[0] < minrange:
        lim_center = (lim[1]+lim[0])/2.
        ax.set_xlim(lim_center-minrange/2., lim_center+minrange/2.)

def limit_ylim(ax, minrange=0.002):
    """Keeps axis reasonable
    ax: axis handle
    minrange: minimum range
    """
    lim = ax.get_ylim()
    if lim[1] - lim[0] < minrange:
        lim_center = (lim[1]+lim[0])/2.
        ax.set_ylim(lim_center-minrange/2., lim_center+minrange/2.)

def plot_timeseries(spks_in, acc_state, spks_out, tmin=None, tmax=None,
                    threshold=None):
    fig, axs = plt.subplots(nrows=3, figsize=(12,9), sharex=True)
    rasterplot(spks_in, axs[0])
    axs[1].plot(acc_state.time, acc_state.state)
    if threshold:
        axs[1].axhline(threshold, color='r', linestyle=":")
        axs[1].axhline(-threshold, color='r', linestyle=":")

    out_weights = np.unique(spks_out.weights)
    if len(out_weights) == 1:
        rasterplot(spks_out, axs[2])
    else:
        spks_out_p = SpikeTrain(
            np.array(spks_out.times)[np.array(spks_out.weights) == 1], 1)
        spks_out_m = SpikeTrain(
            np.array(spks_out.times)[np.array(spks_out.weights) == -1], -1)
        ax = rasterplot([spks_out_p, spks_out_m], axs[2])
        ax.set_yticklabels([1, -1])
        ax.set_ylabel('Weight')

    # set xlims
    axs[-1].set_xlim([0, axs[-1].get_xlim()[1]])
    if tmin:
        axs[-1].set_xlim([tmin, axs[-1].get_xlim()[1]])
    if tmax:
        axs[-1].set_xlim([axs[-1].get_xlim()[0], tmax])
        
    axs[-1].set_xlabel('Time')
    axs[0].set_title('Accumulator Input Spikes')
    axs[0].set_ylabel('Neuron Index')
    axs[1].set_title('Accumulator State')
    axs[2].set_title('Accumulator Output Spikes')

def plot_isi(spks, bins=50, name="Output ", normed=False,
        plot_isi_bool=True, plot_acorr=True, plot_hist=True
    ):
    """Plots the interspike intervals and their autocorrelation and histogram
    
    Creates histogram of all spikes
    and histograms of 
    Separates spikes into +1 and -1 spikes if applicable
    """
    ret = {} # for packing return value
    spks.times = np.array(spks.times)
    spks.weights = np.array(spks.weights)
    isi = np.diff(spks.times)
    if plot_isi_bool:
        fig, ax = plt.subplots(nrows=1, figsize=(12,3))
        ax.plot(isi, 'o')
        ax.set_title(name + 'Interspike Intervals')
        ax.set_xlabel('Interval Index')
        ax.set_ylabel('Interval Duration')
        limit_ylim(ax)
        ret['plot'] = ax

    if plot_acorr:
        isi_centered = isi - np.mean(isi)
        isi_var = np.var(isi)
        if len(np.unique(isi.round(decimals=12))) > 1:
            # compute autocorrelation
            isi_acorr = np.correlate(isi_centered, isi_centered, 'full')/isi_var
            analysis = "Autocorrelation"
        else:
            # if process is deterministic, autocorrelation is undefined
            isi_acorr = np.correlate(isi_centered, isi_centered, 'full') 
            analysis = "Autocovariance"
        shift = np.arange(2*len(isi_centered)-1)-len(isi_centered)
        fig, ax = plt.subplots(nrows=1, figsize=(12,3))
        ax.plot(shift, isi_acorr, 'o')
        ax.set_title(name + 'Interspike Interval ' + analysis)
        ax.set_xlabel('Shift')
        limit_ylim(ax)
    

    if plot_hist:
        if len(np.unique(isi.round(decimals=12))) > 1:
            fig, ax = plt.subplots(nrows=1, figsize=(12,3))
            n, bins, patches = ax.hist(isi, bins, normed=normed)
            ax.set_title(name + 'ISI Histogram (%d ISIs)'%len(isi))
            ax.set_xlabel('Time')
            ax.set_ylabel('Counts')
            ret['hist'] = ax

        # split into +1/-1 weights
        if(len(np.unique(spks.weights)) == 2):
            fig, axs = plt.subplots(nrows=2, figsize=(12,6), sharex=True)        

            isi = np.diff(spks.times[spks.weights == 1])
            n, bins, patches = axs[0].hist(isi, bins, normed=normed)
            axs[0].set_title(name + '+1 Weighted ISI Histogram (%d spikes)'%len(isi))
            axs[0].set_xlabel('Time')
            axs[0].set_ylabel('Counts')
            
            isi = np.diff(spks.times[spks.weights == -1])
            n, bins, patches = axs[1].hist(isi, bins, normed=normed)
            axs[1].set_title(name + '-1 Weighted ISI Histogram (%d spikes)'%len(isi))
            axs[1].set_xlabel('Time')
            axs[1].set_ylabel('Counts')

            ret['split_hist'] = axs
    return ret

def run_regular_neuron_experiment(weight, threshold):
    """Runs a simple experiment with a single neuron"""
    N = 1
    input_rates = 100 # neuron spike rate
    spks_in, acc_state, spks_out = run_neuron_experiment(
        N=N, input_rates=input_rates, weights=weight,
        threshold=threshold, T=0.5
    )
    plot_timeseries(spks_in, acc_state, spks_out, threshold=threshold)
    plot_isi(spks_out, bins=50)

def plot_gamma(ax, shape, scale):
    """plot a gamma distribution on the supplied axis"""
    tmin = max(0, ax.get_xlim()[0])
    tmax = ax.get_xlim()[1]
    t = np.linspace(tmin, tmax)
    ax.plot(t, gamma.pdf(t, a=shape, scale=scale), 'r',
        label="gamma pdf, $shape=%.1f, scale=%f$"%(shape, scale))
    xlims = ax.set_xlim(tmin, tmax)
    return ax

def run_poisson_neuron_experiment(weight, threshold,
        make_timeseries_plot=True, make_isi_plot=True, make_in_isi_plot=True,
        make_out_isi_plot=True,
        T=20, nspikes=None
    ):
    N = 1
    input_rates = 1000 # neuron spike rate
    spks_in, acc_state, spks_out = run_neuron_experiment(
        N=N, input_rates=input_rates, weights=weight, threshold=threshold,
        T=T, nspikes=nspikes, neuron_model=PoissonNeuron
    )
    if make_timeseries_plot:
        plot_timeseries(spks_in, acc_state, spks_out, tmax=0.05,
             threshold=threshold)
    if make_isi_plot:
        if make_in_isi_plot:
            ax = plot_isi(spks_in[0], bins=100, name="Input ", normed=True,
                plot_isi_bool=False, plot_acorr=False)
            tmin = 0.
            tmax = ax['hist'].get_xlim()[1]
            t = np.linspace(tmin, tmax)
            ax['hist'].plot(t, input_rates*np.exp(-input_rates*t), 'r',
                label="exponential pdf, $\lambda=%.0f$"%input_rates)
            ax['hist'].legend(loc='best')
            ax['hist'].set_ylabel('normalized counts')
            ax['hist'].set_xlim(0, tmax)
        ax = plot_isi(spks_out, bins=100, normed=True,
            plot_isi_bool=make_out_isi_plot)
        plot_gamma(ax['hist'], shape=float(threshold)/weight,
            scale=1./input_rates)
        # tmin = 0.
        # tmax = ax['hist'].get_xlim()[1]
        # t = np.linspace(tmin, tmax)
        # k = float(threshold)/weight
        # ax['hist'].plot(t, gamma.pdf(t, a=k, scale=1./input_rates), 'r',
        # label="gamma pdf, $shape=%.1f, scale=1/%.0f$"%(k, input_rates))
        ax['hist'].legend(loc='best')
        ax['hist'].set_ylabel('normalized counts')
        xlims = ax['hist'].set_xlim(0, tmax)
    ret = {'spks_in':spks_in, 'acc_state':acc_state,'spks_out':spks_out}
    return ret

def compute_out_rate(pool, threshold):
    """Computes the output spike rate of an accumulator
    """
    rates = np.array([neuron.spike_rate for neuron in pool.neurons])
    weights = np.array([neuron.weight for neuron in pool.neurons])
    return np.sum(rates*weights)/threshold

