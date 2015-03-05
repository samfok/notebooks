# defines utility functions used in the superfast_network notebook
import numpy as np
from matplotlib import pyplot as plt

import nengo

from neuron.utils.plot import make_blue_cmap, make_red_cmap, make_color_cycle


def square_wave_fun(period=.1, amplitude=.5):
    """Creates a square wave function"""
    return lambda t: amplitude*(2.*(int(t/period) % 2) - 1.)


def build_chain(N_ens=4, conn_syn=.01, probe_syn=.01, N_neurons=100,
                probe_spikes=False):
    """Builds a Network comprising a chain of Ensembles"""
    if not isinstance(conn_syn, (list, np.ndarray)):
        conn_syn = [conn_syn for i in xrange(N_ens-1)]
    if not isinstance(probe_syn, (list, np.ndarray)):
        probe_syn = [probe_syn for i in xrange(N_ens)]
    assert len(conn_syn) == N_ens-1
    assert len(probe_syn) == N_ens
    net = nengo.Network(seed=1)
    net.e = []
    net.c = []
    net.p = []   # ensemble probe
    net.cp = []  # connection probe
    net.sp = []  # spike probe
    with net:
        for i in xrange(N_ens):
            net.e.append(nengo.Ensemble(N_neurons, 1, radius=1., seed=1))
            net.p.append(nengo.Probe(net.e[-1], synapse=probe_syn[i]))
            if probe_spikes:
                net.sp.append(nengo.Probe(net.e[-1].neurons, 'spikes'))
        for i in xrange(N_ens-1):
            net.c.append(
                nengo.Connection(net.e[i], net.e[i+1], synapse=conn_syn[i]))
            net.cp.append(nengo.Probe(net.c[-1], synapse=None))
        net.stim = nengo.Node(None, size_in=1)
        nengo.Connection(net.stim, net.e[0], synapse=None)
    return net


def run_chain_exp(T, N_ens, conn_syn=.01, probe_syn=.01, N_neurons=100,
                  probe_spikes=False, pb=False):
    """Simulates a Network comprising a chain of Ensembles"""
    net = build_chain(N_ens, conn_syn, probe_syn, N_neurons, probe_spikes)
    with net:
        stim = nengo.Node(square_wave_fun(), size_out=1)
        nengo.Connection(stim, net.stim, synapse=None)
    sim = nengo.Simulator(net)
    sim.run(T, progress_bar=pb)
    return net, sim


def plot_decode_input(N_ens, net, sim):
    """Plots the decode and connection inputs of a chain of Ensembles"""
    t = sim.trange()
    bcc = make_color_cycle(N_ens, make_blue_cmap(1., 0.))
    rcc = make_color_cycle(N_ens-1, make_red_cmap(1., 0.))
    fig, axs = plt.subplots(ncols=2, sharex=True, figsize=(12, 4))
    for p_idx, p in enumerate(net.p):
        axs[0].plot(t, sim.data[p], c=bcc[p_idx], alpha=.7)
    axs[0].set_xlabel('time', fontsize=16)
    axs[0].set_ylabel('probed ensemble data', fontsize=16)
    for cp_idx, p in enumerate(net.cp):
        axs[1].plot(t, sim.data[p], c=rcc[cp_idx], alpha=.7)
    axs[1].set_xlabel('time', fontsize=16)
    axs[1].set_ylabel('probed connection data', fontsize=16)
    return fig, axs
