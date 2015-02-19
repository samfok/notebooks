import numpy as np
from matplotlib import pyplot as plt
from neuron import (
    th_lif_fi, num_rate_alif_fi, run_ralifsoma, th_lif_dfdu)
from plot import make_blue_cmap, make_red_cmap, make_color_cycle


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
    num_f = num_rate_alif_fi(u_in, tau_m, tref, xt, af, tau_f)
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
    n_steps = int(np.ceil(T/dt))
    u_in = u_in * np.ones(n_steps)
    f, u_net = run_ralifsoma(dt, u_in, tau_m, tref, xt, af, tau_f, f0, u0)
    return f, u_net


def add_traj(ax, u_in, tau_m, tref, xt, af, tau_f, dt=1e-4,
             T=None, u0=None, f0=None):
    f, u_net = u_in_traj(u_in, tau_m, tref, xt, af, tau_f,
                         dt=dt, T=T, u0=u0, f0=f0)
    ax.plot(u_net, f, 'o-m')


def u_in_gain(tau_m, tref, xt, af):
    max_u = 5.
    u_in = np.array(
        [0.] + np.logspace(np.log10(xt), np.log10(2.*xt), 20).tolist() +
        np.linspace(2.*xt, max_u, 10).tolist())
    idx = u_in > xt
    fig, ax_f = plt.subplots(figsize=(8, 6))
    ax_g = ax_f.twinx()
    title_str = r'$\tau_m=%.3f$ $t_{ref}=%.3f$ ' % (tau_m, tref)

    f_open = th_lif_fi(u_in, tau_m, tref, xt)
    open_line, = ax_f.plot(u_in, f_open, 'k', lw=2)

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
        f_closed = num_rate_alif_fi(u_in, tau_m, tref, xt, af_val)
        gain = np.zeros_like(u_in)
        gain[idx] = f_open[idx] / f_closed[idx]
        label_str = r'$\alpha_f=%.2f$' % (af_val)
        ax_f.plot(u_in, f_closed, c=cc_b[i], alpha=.7, label=label_str)
        ax_g.semilogy(u_in[idx], gain[idx],
                      c=cc_r[i], alpha=.7, label=label_str)

    try:
        ax_f.set_ylim(0, 1./tref)
    except ZeroDivisionError:
        ax_f.set_ylim(0, 1.1*np.max(f_open))
    ax_f.set_xlabel(r'$u_{in}$', fontsize=20)
    ax_f.set_ylabel(r'$f$', fontsize=20, rotation=0)
    ax_g.set_ylabel(r'$A$', fontsize=20, rotation=0)
    fopen_legend = plt.legend((open_line,), ('LIF',), loc='lower left')
    ax_f.add_artist(fopen_legend)
    if n == 1:
        title_str += label_str
    else:
        ax_f.legend(loc='upper right', bbox_to_anchor=(.6, 1.))
        ax_g.legend(loc='upper left', bbox_to_anchor=(.6, 1.))
    ax_f.set_title(title_str, fontsize=14)
