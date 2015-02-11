import numpy as np
import matplotlib.pyplot as plt
from neuron import th_lif_fi, th_lif_if, num_rate_alif_fi, run_ralifsoma
from plot import make_blue_cmap, make_red_cmap, make_color_cycle


def phase_uin_uf(tau_m=.01, tref=.005, xt=1., af=.02, tau_f=.005,
                 max_u=5., n_f=15, dt=1e-3):
    """Generates two phase plots of u_in vs f and u_f vs f
    
    Returns
    -------
    A figure handle containing the axes for the two phase plots
    The u_in along which the phase plots were calculated
    """
    inc = -np.expm1(-dt/tau_f)

    u = np.sort(np.linspace(0, max_u, 20).tolist() + [xt, 1.001*xt])
    f = th_lif_fi(u, tau_m, tref, xt)
    max_f = max(f)
    min_f = max(f)/10.
    ralif_f = num_rate_alif_fi(u, tau_m, tref, xt, af, tau_f)

    fig = plt.figure(figsize=(16, 6))
    ax = fig.add_subplot(121)
    axes = [ax]
    ax.plot(u, f, 'k', lw=1, label=r'LIF')
    ax.plot(u, ralif_f, 'c', lw=2, label='adaptive rate LIF')

    f, f_step = np.linspace(min_f, max_f, n_f, retstep=True)

    U_in, F, dU_in, dF = [], [], [], []
    for f_open in f:
        f_arr = np.arange(f_open, 0., -f_step)
        u_in_arr = th_lif_if(f_open, tau_m, tref, xt)*np.ones(f_arr.shape)
        u_arr = th_lif_if(f_arr, tau_m, tref, xt)
        u_f_arr = u_in_arr - u_arr
        du_f_arr = -inc*u_f_arr + inc*af*f_arr
        df_arr = th_lif_fi(u_in_arr-(u_f_arr+du_f_arr), tau_m, tref, xt)-f_arr

        U_in += u_in_arr.tolist()
        F += f_arr.tolist()
        dU_in += np.zeros_like(u_in_arr).tolist()
        dF += df_arr.tolist()

    bcmap = make_blue_cmap(low=1., high=0.)
    ax.quiver(U_in, F, dU_in, dF, dF,
              cmap=bcmap, alpha=.7, angles='xy', pivot='middle')
    ax.set_xlim(0, max_u)
    ax.legend(loc='upper left')
    ax.set_xlabel(r'$u_{in}$', fontsize=20)
    ax.set_ylabel(r'$f$', fontsize=20)

    u_in = th_lif_if(f, tau_m, tref, xt)
    rcmap = make_red_cmap()
    cc = make_color_cycle(range(n_f), rcmap)
    for idx, (u_val, f_val) in enumerate(zip(u_in, f)):
        ax.plot([u_val, u_in[-1]], [f_val, f_val], c=cc[idx])
        ax.plot([u_val, u_val], [0, f_val], c=cc[idx], ls=':')

    U_f, F, dU_f, dF = [], [], [], []

    n_uf_fine = 30
    u_f_fine = np.zeros((n_uf_fine, n_f))
    for idx, u_val in enumerate(u_in):
        u_f_fine[:, idx] = np.linspace(0, u_val-xt, n_uf_fine)
    f_fine = th_lif_fi(u_in-u_f_fine, tau_m, tref, xt)
    ax = fig.add_subplot(122, sharey=ax)
    axes.append(ax)
    ax.set_color_cycle(cc)
    lines = ax.plot(u_f_fine, f_fine)
    lines[0].set_label(r'$u_{in}=%.2f$' % (u_in[0]))
    lines[-1].set_label(r'$u_{in}=%.2f$' % (u_in[-1]))

    for idx, u_val in enumerate(u_in):
        f_open = th_lif_fi(u_val, tau_m, tref, xt)
        f_arr = np.arange(f_open, 0., -f_step)
        u_net = th_lif_if(f_arr, tau_m, tref, xt)
        u_f_arr = u_val - u_net
        du_f_arr = -inc*u_f_arr + af*inc*f_arr
        df_arr = th_lif_fi(u_val-(u_f_arr+du_f_arr), tau_m, tref, xt)-f_arr

        U_f += u_f_arr.tolist()
        F += f_arr.tolist()
        dU_f += du_f_arr.tolist()
        dF += df_arr.tolist()
    ax.quiver(U_f, F, dU_f, dF, dF,
              cmap=bcmap, alpha=.7, angles='xy', pivot='middle')

    u_f = np.array([0, max_u])
    f = u_f / af
    ax.plot(u_f, f, 'c', label=r'$u_f=\alpha_ff$')

    ralif_f = num_rate_alif_fi(u_in, tau_m, tref, xt, af, tau_f)
    u_f = af*ralif_f
    ax.plot(u_f, ralif_f, 'co', alpha=1.)
    axes[0].plot(u_in, ralif_f, 'co', alpha=1.)

    ax.set_xlim(0, max_u)
    ax.set_ylim(0, 1./tref)
    ax.legend(loc='upper left', fontsize=16)
    ax.set_xlabel(r'$u_f$', fontsize=20)

    return fig, u_in
