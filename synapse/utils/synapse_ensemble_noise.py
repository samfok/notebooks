# utils for synapse_ensemble_noise notebook
import numpy as np
from matplotlib import pyplot as plt
import nengo

from synapse import th_u_var, th_p_var, th_u_xmax, th_u_xmin
from neuron.utils.plot import make_color_cycle


def build_net(N=10, u=.5, tau=.01):
    """builds a single ensemble network"""
    net = nengo.Network(seed=1)
    with net:
        net.stim = nengo.Node(u)
        net.ens = nengo.Ensemble(N, 1, seed=1)
        net.out = nengo.Node(lambda t, x: x, size_in=1)
        nengo.Connection(net.stim, net.ens, synapse=None)
        net.conn = nengo.Connection(net.ens, net.out, synapse=tau,
                                    solver=nengo.solvers.LstsqL2(reg=0.1))
        net.p = nengo.Probe(net.out, synapse=None)
        net.prates = nengo.Probe(net.ens.neurons, 'spikes', synapse=tau)
    return net


def check_decode_var(dt, T, N, u, tau):
    """Compares the observed variance with the predicted variance"""
    net = build_net(N, u, tau)
    sim = nengo.Simulator(net, dt=dt)
    sim.run(T, progress_bar=False)

    encoders = sim.data[net.ens].encoders.reshape(-1)
    decoders = sim.data[net.conn].decoders.reshape(-1)
    gain = sim.data[net.ens].gain
    bias = sim.data[net.ens].bias
    f = net.ens.neuron_type.rates(u*encoders, gain, bias)
    f_idx = f > 0
    syn_xmax = th_u_xmax(f[f_idx], tau)
    syn_xmin = th_u_xmin(f[f_idx], tau)

    t = sim.trange()
    decode = sim.data[net.p].reshape(-1)
    rates = sim.data[net.prates]
    decode_from_rates = np.sum(rates * decoders, axis=1)

    # plot decode and spike rates
    N_plot = 2
    max_T = min(.5, T)
    cc = make_color_cycle(N_plot, plt.cm.Set1)

    t_idx = t <= max_T
    fig, axs = plt.subplots(ncols=2, figsize=(14, 4))
    axs[0].plot(t[t_idx], decode[t_idx], 'b', lw=2, label='observed decode')
    axs[0].plot(t[t_idx], decode_from_rates[t_idx],
                'r', label='decoders * rates')
    axs[0].legend(loc='lower right')
    axs[0].set_ylabel('decode', fontsize=12)
    axs[0].set_xlabel('time', fontsize=12)
    axs[1].set_color_cycle(cc)
    axs[1].plot(t[t_idx], rates[t_idx, :N_plot], alpha=.8)
    axs[1].plot(axs[1].get_xlim(), [f[f_idx][:N_plot], f[f_idx][:N_plot]],
                alpha=.8)
    axs[1].plot(axs[1].get_xlim(), [syn_xmax[:N_plot], syn_xmax[:N_plot]],
                ':', alpha=.8)
    axs[1].plot(axs[1].get_xlim(), [syn_xmin[:N_plot], syn_xmin[:N_plot]],
                ':', alpha=.8)
    axs[1].set_ylabel('rates', fontsize=12)
    axs[1].set_xlabel('time', fontsize=12)
    axs[1].set_title('first %d spiking neurons, first %.2fs' % (N_plot, max_T))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(decoders, 50, histtype='step')
    ax.set_xlabel('decode weights', fontsize=12)

    t_idx = t > 10*tau
    var_decode = np.var(decode[t_idx])
    std_decode = np.sqrt(var_decode)

    var_decode_from_rates = np.var(decode_from_rates[t_idx])
    std_decode_from_rates = np.sqrt(var_decode_from_rates)

    var_rates = np.var(rates[t_idx], axis=0)

    est_var_decode = np.sum((decoders)**2 * var_rates)
    est_std_decode = np.sqrt(est_var_decode)

    f_idx = f > 0
    pred_var_rates = np.zeros_like(f)
    pred_var_rates_p = np.zeros_like(f)
    pred_var_rates[f_idx] = th_u_var(f[f_idx], tau)  # uniform assumption
    pred_var_rates_p[f_idx] = th_p_var(f[f_idx], tau)  # poisson assumption
    pred_var_decode = np.sum(decoders**2 * pred_var_rates)
    pred_var_decode_p = np.sum(decoders**2 * pred_var_rates_p)
    pred_std_decode = np.sqrt(pred_var_decode)
    pred_std_decode_p = np.sqrt(pred_var_decode_p)

    var_rel_error = np.abs(var_decode - pred_var_decode) / var_decode

    print('observed  decode variance %e' % var_decode)
    print('decoders * rates variance %e' % var_decode_from_rates)
    print('estimated decode variance %e' % (est_var_decode) +
          ' (from observed rates)')
    print('predicted decode variance %e' % pred_var_decode +
          ' (from theoretical rates, assume uniform spikes)')
    print('predicted decode variance %e' % pred_var_decode_p +
          ' (from theoretical rates, assume Poisson spikes)')
    print('relative variance error   %e' % var_rel_error +
          ' (observed variance vs var predicted from decoding uniform rates)')
    print
    print('observed  decode std %e' % std_decode)
    print('decoders * rates std %e' % std_decode_from_rates)
    print('estimated decode std %e (from observed rates)' % est_std_decode)
    print('predicted decode std %e' % pred_std_decode +
          ' (from theoretical rates, assuming uniform spikes)')
    print('predicted decode std %e' % pred_std_decode_p +
          ' (from theoretical rates, assuming Poisson spikes)')
