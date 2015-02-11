import numpy as np
import matplotlib.pyplot as plt
from neuron import th_lif_fi, th_lif_if, num_rate_alif_fi, run_ralifsoma
from plot import make_blue_cmap, make_red_cmap, make_color_cycle


def phase_uin_uf(tau_m, tref, xt, af, tau_f, dt,
                 max_u=5., n_f=15):
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
    min_f = max_f/10.
    ralif_f = num_rate_alif_fi(u, tau_m, tref, xt, af, tau_f)

    fig = plt.figure(figsize=(16, 6))
    ax = fig.add_subplot(121)
    ax.plot(u, f, 'k', lw=2, label='nonadaptive LIF')
    ax.plot(u, ralif_f, 'c', lw=2, label='adaptive rate LIF')

    f, f_step = np.linspace(min_f, max_f, n_f, retstep=True)

    U_in, F, dU_in, dF = [], [], [], []
    for f_open in f:
        f_arr = np.arange(f_open, min_f-1e-8, -f_step)
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
              cmap=bcmap, alpha=.7, angles='xy')
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

    # uf phase plot
    n_uf_fine = 30
    u_f_fine = np.zeros((n_uf_fine, n_f))
    for idx, u_val in enumerate(u_in):
        u_f_fine[:, idx] = np.sort(
            np.linspace(0, u_val-xt, n_uf_fine-1).tolist() + [u_val-1.001*xt])
    f_fine = th_lif_fi(u_in-u_f_fine, tau_m, tref, xt)
    ax = fig.add_subplot(122, sharey=ax)
    ax.set_color_cycle(cc)
    lines = ax.plot(u_f_fine, f_fine)
    lines[0].set_label(r'$u_{in}=%.2f$' % (u_in[0]))
    lines[-1].set_label(r'$u_{in}=%.2f$' % (u_in[-1]))

    U_f, F, dU_f, dF = [], [], [], []
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
              cmap=bcmap, alpha=.7, angles='xy')

    u_f = np.array([0, max_u])
    f = u_f / af
    ax.plot(u_f, f, 'c', label=r'$u_f=\alpha_ff$')

    ralif_f = num_rate_alif_fi(u_in, tau_m, tref, xt, af, tau_f)
    u_f = af*ralif_f
    ax.plot(u_f, ralif_f, 'co', alpha=1.)
    fig.axes[0].plot(u_in, ralif_f, 'co', alpha=1.)

    ax.set_xlim(0, max_u)
    ax.set_ylim(0, 1./tref)
    ax.legend(loc='upper left', fontsize=16)
    ax.set_xlabel(r'$u_f$', fontsize=20)

    return fig, u_in


def phase_u_uf(tau_m, tref, xt, af, tau_f, dt,
               max_u=5., u_in=3.5):
    """Generates two phase plots of u vs f and u_f vs f where u = u_in - uf

    Returns
    -------
    A figure handle containing the axes for the two phase plots
    """
    inc = -np.expm1(-dt/tau_f)
    f_open = th_lif_fi(u_in, tau_m, tref, xt)
    u = np.sort(np.linspace(0, max_u, 30).tolist() + [xt, 1.001*xt])
    f = th_lif_fi(u, tau_m, tref, xt)
    max_f = max(f)
    min_f = max_f/20.

    # u phase plot
    fig = plt.figure(figsize=(16,6))
    ax = fig.add_subplot(121)
    ax.plot(u, f, 'k', lw=2, label=r'$u_f=0$')
    ax.plot([u_in, max_u], [f_open, f_open], 'r')
    ax.plot([u_in, u_in], [0., f_open], 'r:')

    # interior of u phase plot
    U, F, dU, dF = [], [], [], []
    f, f_step = np.linspace(max_f, min_f, 20, retstep=True)
    for f_val in f:
        f_arr = np.arange(f_val, min_f-1e-8, f_step)
        u = th_lif_if(f_arr, tau_m, tref, xt)
        u_arr = u[0] * np.ones(f_arr.shape)
        uf = u_in - u_arr
        duf = -inc*uf + inc*af*f_arr
        df_arr = th_lif_fi(u_in-(uf+duf), tau_m, tref, xt) - f_arr
        du_arr = -duf

        U += u_arr.tolist()
        F += f_arr.tolist()
        dU += du_arr.tolist()
        dF += df_arr.tolist()
    bcmap = make_blue_cmap()
    ax.quiver(U, F, dU, dF, dF,
              cmap=bcmap, alpha=.7, angles='xy')

    # border of u phase plot
    f = f[:]
    u = th_lif_if(f, tau_m, tref, xt)
    uf = u_in - u
    duf = -inc*uf + inc*af*f
    df = th_lif_fi(u_in-(uf+duf), tau_m, tref, xt) - f
    du = -duf
    rcmap = make_red_cmap()
    ax.quiver(u, f, du, df, alpha=.8, angles='xy')

    f_num = num_rate_alif_fi(u_in, tau_m, tref, xt, af, tau_f)
    u_num = th_lif_if(f_num, tau_m, tref, xt)
    ax.plot(u_num, f_num, 'co')
    ax.legend(loc='upper left', fontsize=16)
    ax.set_xlim(0, max_u)
    ax.set_xlabel(r'$u$', fontsize=20)
    ax.set_ylabel(r'$f$', fontsize=20)

    # uf phase plot
    u_f_fine = np.sort(np.linspace(0, u_in-xt, 30).tolist() + [u_in-1.001*xt])
    u_fine = u_in - u_f_fine
    f_fine = th_lif_fi(u_fine, tau_m, tref, xt)
    ax = fig.add_subplot(122, sharey=ax)
    ax.plot(u_f_fine, f_fine, c='r', label=r'$u_{in}=%.2f$' % u_in)
    
    U_f, F, dU_f, dF = [], [], [], []
    f_range = f_open/5.
    f_major_step = f_open/15.
    f_minor_step = f_range/2.
    f = np.arange(f_open, min_f, -f_major_step)
    for f_val in f:
        u = th_lif_if(f_val, tau_m, tref, xt)
        u_f = u_in - u
        f_low = max(0, f_val-f_range)
        f_high = f_val+f_range
        f_arr = np.arange(f_low, f_high+1e-8, f_minor_step)
        u_f_arr = u_f * np.ones(f_arr.shape)
        du_f_arr = -inc*u_f_arr + inc*af*f_arr
        df_arr = th_lif_fi(u_in-(u_f_arr+du_f_arr), tau_m, tref, xt)-f_arr
    
        U_f += u_f_arr.tolist()
        F += f_arr.tolist()
        dU_f += du_f_arr.tolist()
        dF += df_arr.tolist()
    
    ax.quiver(U_f, F, dU_f, dF, dF,
              cmap=bcmap, alpha=.7, angles='xy', width=.004)
    u_f = np.array([0, max_u])
    f = u_f / af
    ax.plot(u_f, f, 'c', label=r'$u_f=\alpha_ff$')
    ax.plot(u_in - u_num, f_num, 'co')
    ax.legend(loc='upper right', fontsize=16)
    ax.set_xlim(0, u_in)
    ax.set_ylim(0, 1.1*max_f)
    ax.set_xlabel(r'$u_f$', fontsize=20)
    return fig


def u_in_traj(u_in, tau_m, tref, xt, af, tau_f, dt=1e-3, T=None, f0=0.):
    """Generates a trajectory resulting from u_in
    
    Returns
    -------
    f : rates
    uf: adaptive feedback
    """
    if T is None:
        T = 5.*tau_f
    n_steps = int(np.ceil(T/dt))
    u_in = u_in * np.ones(n_steps)
    f, uf = run_ralifsoma(dt, u_in, tau_m, tref, xt, af, tau_f, f0)
    return f, uf
