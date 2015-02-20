import numpy as np
from matplotlib import pyplot as plt
from neuron import (
    th_lif_fi, th_lif_if, th_lif_dfdu, th_ralif_if,
    num_alif_fi, num_ralif_fi, run_ralifsoma,
    run_lifsoma, run_alifsoma)
from plot import make_blue_cmap, make_red_cmap, make_color_cycle
from signal import filter_spikes
from nengo.synapses import filt
from data import scalar_to_array


def phase_u_f(tau_m, tref, xt, af, tau_f, dt=1e-4, max_u=5., u_in=3.5,
              show=False):
    """Generates a phase plot of u vs """
    def th_du_dt(u_net, u_in, f, af, tau_f):
        return 1./tau_f * (-u_net + u_in - af*f)

    min_f = 10.
    _u = np.sort(np.linspace(0, max_u, 100).tolist() + [xt, xt*1.001])
    _f = th_lif_fi(_u, tau_m, tref, xt)
    max_f = 1./tref

    u_vals = np.linspace(0., max_u, 20)
    f_step = max_f/20.

    U, F, dUdt, dFdt = [], [], [], []
    for u_val in u_vals:
        f_open = max_f
        f = np.array(np.arange(min_f, f_open, f_step).tolist() + [f_open])
        u = u_val * np.ones(f.shape)
        dudt = th_du_dt(u, u_in, f, af, tau_f)
        dfdu = th_lif_dfdu(u, tau_m, tref, xt)
        dfdt = dfdu*dudt

        U += u.tolist()
        F += f.tolist()
        dUdt += dudt.tolist()
        dFdt += dfdt.tolist()

    f_ss = (u_in - _u) / af
    num_f = num_ralif_fi(u_in, tau_m, tref, xt, af)
    uf = af*num_f
    num_u = u_in - uf

    bcmap = make_blue_cmap()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(_u, _f, 'k')
    ax.quiver(U, F, dUdt, dFdt, dFdt, angles='xy', scale_units='xy',
              pivot='middle', cmap=bcmap, alpha=.7)
    ax.plot(_u, f_ss, 'c')
    ax.plot(num_u, num_f, 'co')

    ax.set_xlim(0, max_u)
    ax.set_ylim(0, 1./tref)
    ax.set_xlabel(r'$u_{net}$', fontsize=20)
    ax.set_ylabel(r'$f$', fontsize=20)

    if show:
        plt.show()
    return fig, ax


def u_in_traj(u_in, tau_m, tref, xt, af, tau_f,
              dt=1e-4, T=None, u0=None, f0=None):
    """Generates a trajectory given a fixed u_in

    Parameters
    ----------
    u_in : array-like (m x n)
        inputs to the m neurons for each of the n time steps
    tau_m : float
        soma time constant (s)
    xt : float
        threshold
    af : float (optional)
        scales the feedback synapse state into a current
    tau_f : float (optional)
        time constant of the feedback synapse
    dt : float (optional)
        time step (s)
    T : float (optional)
        total simulation time, defaults to 5 * tau_f
    u0 : array-like (m,) or float (optional)
        initial net input to the neurons
    f0 : array-like (m,) or float (optional)
        initial rates of the neurons

    Returns
    -------
    f : numpy array (m x n)
        neuron rates
    u_net : numpy array (m x n)
        net input to neurons
    """
    if T is None:
        T = 5.*tau_f
    if u0 is None:
        u0 = u_in
    if f0 is None:
        f0 = th_lif_fi(u0, tau_m, tref, xt)
    N = int(np.ceil(T/dt))
    u_in = u_in * np.ones(N)
    f, u_net = run_ralifsoma(
        dt, u_in, tau_m, tref, xt, af, tau_f, f0, u0, ret_u=True)
    return f, u_net


def add_traj(ax, u_in, tau_m, tref, xt, af, tau_f, dt=1e-4,
             T=None, u0=None, f0=None):
    f, u_net = u_in_traj(u_in, tau_m, tref, xt, af, tau_f,
                         dt=dt, T=T, u0=u0, f0=f0)
    ax.plot(u_net, f, 'o-m')


def u_in_gain(tau_m, tref, xt, af):
    """Computes relative gain between open and closed loop tuning curves"""
    max_u = 5.
    u_in = np.array(
        [0.] + np.logspace(np.log10(xt), np.log10(1.5*xt), 30).tolist() +
        np.linspace(1.5*xt, max_u, 10).tolist())
    idx = u_in > xt
    fig, ax_f = plt.subplots(figsize=(8, 6))
    ax_g = ax_f.twinx()
    title_str = r'$\tau_m=%.3f$ $t_{ref}=%.3f$ ' % (tau_m, tref)

    f_open = th_lif_fi(u_in, tau_m, tref, xt)
    oline, = ax_f.plot(u_in, f_open, 'k', lw=2)

    if isinstance(af, (list, np.ndarray)):
        assert isinstance(af, (list, np.ndarray))
        n = len(af)
        rcmap = make_red_cmap(.5)
        bcmap = make_blue_cmap(.2)
        cc_r = make_color_cycle(np.arange(n), rcmap)
        cc_b = make_color_cycle(np.arange(n), bcmap)
    else:
        n = 1
        af = [af]
        cc_r = ['r']
        cc_b = ['b']
    for i, af_val in enumerate(af):
        f_closed = num_ralif_fi(u_in, tau_m, tref, xt, af_val)
        gain = np.zeros_like(u_in)
        gain[idx] = f_open[idx] / f_closed[idx]
        label_str = r'$\alpha_f=%.2f$' % (af_val)
        cline, = ax_f.plot(
            u_in, f_closed, c=cc_b[i], alpha=.7, label=label_str)
        gline, = ax_g.semilogy(u_in[idx], gain[idx],
                               c=cc_r[i], alpha=.7, label=label_str)

    try:
        ax_f.set_ylim(0, 1./tref)
    except ZeroDivisionError:
        ax_f.set_ylim(0, 1.1*np.max(f_open))
    ax_f.set_xlabel(r'$u_{in}$', fontsize=20)
    ax_f.set_ylabel(r'$f$', fontsize=20, rotation=0)
    ax_g.set_ylabel(r'$A$', fontsize=20, rotation=0)
    if n == 1:
        title_str += label_str
        fopen_legend = plt.legend(
            (oline, cline, gline), ('LIF', 'raLIF', r'$A$'), loc='lower left')
    else:
        fopen_legend = plt.legend((oline,), ('LIF',), loc='lower left')
    ax_f.add_artist(fopen_legend)
    if n > 1:
        ax_f.legend(loc='upper right', bbox_to_anchor=(.7, 1.))
        ax_g.legend(loc='upper left', bbox_to_anchor=(.7, 1.))
    ax_f.set_title(title_str, fontsize=14)


def rate_v_spiking(alif_u_in, tau_m, tref, xt, af, tau_f, tau_syn, dt, T=None,
                   normalize=False):
    """Compares the raLIF aLIF, and LIF soma outputs

    Input to the raLIF and LIF neurons adjusted to so their steady-state output
    matches the aLIF steady-state output

    Parameters
    ----------
    alif_u_in : array-like (m x n)
        inputs for each time step
    """
    if T is None:
        T = 5. * tau_f
    N = int(np.ceil(T/dt))
    u_in = alif_u_in + np.zeros(N)

    open_lif_rate = th_lif_fi(u_in[0], tau_m, tref, xt)
    alif_ss_rate = num_alif_fi(u_in[0], tau_m, tref, xt, af, tau_f)

    ralif_u_in_val = th_ralif_if(alif_ss_rate, tau_m, tref, xt, af)
    ralif_u_in = ralif_u_in_val + np.zeros(N)

    lif_u_in_val = th_lif_if(alif_ss_rate, tau_m, tref, xt)
    lif_u_in = lif_u_in_val + np.zeros(N)

    alif_spike_times = run_alifsoma(
        dt, u_in, tau_m, tref, xt, af, tau_f)
    t, alif_rates = filter_spikes(
        dt, T, alif_spike_times, tau_syn, ret_time=True)

    lif_spike_times = run_lifsoma(
        dt, lif_u_in, tau_m, tref, xt)
    lif_rates = filter_spikes(
        dt, T, lif_spike_times, tau_syn, ret_time=False)

    ralif_rates = run_ralifsoma(
        dt, ralif_u_in, tau_m, tref, xt, af, tau_f,
        f0=open_lif_rate, u0=ralif_u_in[0])
    ralif_rates = filt(ralif_rates, tau_syn, dt)

    ylabel_str = r'$x_{syn}$'
    if normalize:
        ralif_rates /= alif_ss_rate
        alif_rates /= alif_ss_rate
        lif_rates /= alif_ss_rate
        alif_ss_rate = 1.
        ylabel_str = r'$x_{syn}/\langle x_{syn, ss}\rangle$'

    alif_tspk0 = alif_spike_times[0]
    idx0 = np.argmax(t > alif_tspk0)-1
    fig, ax = plt.subplots(figsize=(16, 3))
    ax.axhline(alif_ss_rate, ls=':', c='k', alpha=.5)

    ax.plot(t[idx0:], ralif_rates[:-idx0], 'k', alpha=1., label='raLIF')
    ax.plot(t, alif_rates, 'r', alpha=.9, label='aLIF')
    ax.plot(t, lif_rates, 'b', alpha=.6, label='LIF')

    ax.legend(loc='best')
    ax.set_xlabel(r'$t$', fontsize=20)
    ax.set_ylabel(ylabel_str, fontsize=20)
    title_str = (r'$u_{in}=%.2f,%.2f,%.2f$ (aLIF, raLIF, LIF), ' %
                 (u_in[0], ralif_u_in[0], lif_u_in[0], ))
    if tau_f != tau_syn:
        title_str += r'$\tau_f=%.3f$, $\tau_{syn}=%.3f$' % (tau_f, tau_syn)
    else:
        title_str += r'$\tau_f=\tau_{syn}=%.3f$' % (tau_syn)
    ax.set_title(title_str, fontsize=20, y=1.05)
    return fig, ax


def f_traj(u_ins, tau_m, tref, xt, af, tau_f, dt=1e-4, T=None,
           normalizey=False):
    """Simulates an trajectory of the ralif rates"""
    u_ins = scalar_to_array(u_ins)
    if T is None:
        T = 3.*tau_f
    N = int(np.ceil(T/dt))
    t = np.arange(N) * dt

    fig, ax = plt.subplots(figsize=(12, 6))
    rcmap = make_red_cmap()
    cc = make_color_cycle(np.arange(len(u_ins)), rcmap)
    lines = []
    for idx, u_in_val in enumerate(u_ins):
        u_in = u_in_val + np.zeros(N)
        u0 = u_in_val
        f0 = th_lif_fi(u_in_val, tau_m, tref, xt)
        f, _ = u_in_traj(u_in, tau_m, tref, xt, af, tau_f, dt, T, u0, f0)

        if normalizey:
            fss = num_ralif_fi(u_in_val, tau_m, tref, xt, af)
            f /= fss
        lines.append(ax.plot(
            t/tau_f, f, c=cc[idx], label=r'$u_{in}=%.2f$' % u_in_val)[0])
    ax.legend(
        handles=lines[::-1], loc='upper left', bbox_to_anchor=(1.01, 1.02))
    ax.set_xlabel(r'$t/\tau_f$', fontsize=20)
    if normalizey:
        ylabel_str = r'$f/f_{ss}$'
        ax.set_ylim(1., ax.get_ylim()[1])
    else:
        ylabel_str = r'$f$'
    ax.set_ylabel(ylabel_str, fontsize=20)
    return fig, ax
