import numpy as np
from matplotlib import pyplot as plt
from neuron import (
    th_lif_fi, th_lif_if, th_lif_dfdu, th_ralif_if, th_ralif_dfdt,
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
    ax.plot(_u, _f, 'k', label=r'$f_{open}$')
    ax.quiver(U, F, dUdt, dFdt, dFdt, angles='xy', scale_units='xy',
              pivot='middle', cmap=bcmap, alpha=.7)
    ax.plot(_u, f_ss, 'c', label=r'$f=\frac{u_{in}-u_{net}}{\alpha_f}$')
    ax.plot(num_u, num_f, 'co')

    ax.set_xlim(0, max_u)
    ax.set_ylim(0, 1./tref)
    ax.set_xlabel(r'$u_{net}$', fontsize=20)
    ax.set_ylabel(r'$f$', fontsize=20)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1.025), fontsize=18)

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

    ax.plot(t[idx0:], ralif_rates[:-idx0], 'r', alpha=1., label='raLIF')
    ax.plot(t, alif_rates, 'k', alpha=.6, label='aLIF')
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


def syn_out(t, dfdt, tau_syn, f0, fss):
    tss = (fss - f0) / dfdt
    hf = np.zeros_like(t)

    def tran(t, dfdt, tau_syn, f0):
        return dfdt * t + (f0 - dfdt * tau_syn) * (1-np.exp(-t/tau_syn))
    idx = t <= tss
    hf[idx] = tran(t[idx], dfdt, tau_syn, f0)
    hf_tss = tran(tss, dfdt, tau_syn, f0)
    idx = t > tss
    hf[idx] = (fss*(1-np.exp(-(t[idx]-tss)/tau_syn)) +
               hf_tss*np.exp(-(t[idx]-tss)/tau_syn))
    return hf


def f_traj(u_ins, tau_m, tref, xt, af, tau_f, dt=1e-4, T=None,
           tau_syn=None, normalize_y=False):
    """Simulates an trajectory of the ralif rates

    Parameters
    ----------
    tau_syn : float (optional)
        if passed in, also computes and shows filtered rates
    normalize_y : bool (optional)
        Normalize the output rates (y-axis) to the steady-state rates
    """
    u_ins = scalar_to_array(u_ins)
    N_u_in = len(u_ins)
    if T is None:
        T = 3.*tau_f
    N = int(np.ceil(T/dt))
    t = np.arange(N) * dt

    fig, ax = plt.subplots(figsize=(12, 6))
    rcmap = make_red_cmap(1., 0.)
    bcmap = make_blue_cmap(1., 0.)
    rcc = make_color_cycle(np.arange(N_u_in), rcmap)
    bcc = make_color_cycle(np.arange(N_u_in), bcmap)
    lines = []
    for idx, u_in_val in enumerate(u_ins):
        u_in = u_in_val + np.zeros(N)
        u0 = u_in_val
        f0 = th_lif_fi(u_in_val, tau_m, tref, xt)
        fss = num_ralif_fi(u_in_val, tau_m, tref, xt, af)
        f, _ = u_in_traj(u_in, tau_m, tref, xt, af, tau_f, dt, T, u0, f0)
        tau = tau_f / (tau_m*xt) * (u0*(u0 - xt))
        dfdt0 = -af * f0**3 / tau
        f_est = dfdt0 * t + f0
        f_est[f_est < fss] = fss

        if normalize_y:
            f /= fss
            f_est /= fss

        lines.append(ax.plot(
            t/tau_f, f, c=rcc[idx], label='rate')[0])
        ax.plot(t/tau_f, f_est, c=rcc[idx], alpha=.5, label='piecewise rate')
        if normalize_y:
            ax.axhline(1, c='k', ls=':', alpha=.5)
        else:
            ax.axhline(fss, c='k', ls=':', alpha=.5)

        if tau_syn is not None:
            hf = np.zeros(N)

            hf[1:] = filt(f[:-1], tau_syn, dt)
            hf_est = syn_out(t, dfdt0, tau_syn, f0, fss)
            hfss = fss * (1 - np.exp(-t/tau_syn))

            if normalize_y:
                hf_est /= fss
                hfss /= fss

            ax.plot(t/tau_f, hf, c=bcc[idx], label='filtered rate')
            ax.plot(t/tau_f, hf_est, c=bcc[idx], alpha=.5,
                    label='filtered piecewise rate')
            ax.plot(t/tau_f, hfss, c=bcc[idx], ls=':', alpha=.5,
                    label='filtered steady state rate')

    title_str = r'$\tau_m=%.3f$  $t_{ref}=%.3f$  $\alpha_f=%.3f$  ' % (
        tau_m, tref, af)
    if tau_syn:
        title_str += r'$\tau_{syn}/\tau_f=%.3f$  ' % (tau_syn/tau_f)
    if N_u_in > 1:
        ax.legend(
            lines[::-1], [r'$u_{in}=%.3f$' % u for u in u_ins[::-1]],
            loc='upper left', bbox_to_anchor=(1.01, 1.02))
    else:
        title_str += r'$u_{in}=%.3f$  ' % (u_ins)
        ax.legend(loc='upper right')
    ax.set_xlabel(r'$t/\tau_f$', fontsize=20)
    if normalize_y:
        ylabel_str = r'$f/f_{ss}$'
        if tau_syn is None:
            ax.set_ylim(1., ax.get_ylim()[1])
    else:
        ylabel_str = r'$f$'
    ax.set_ylabel(ylabel_str, fontsize=20)
    ax.set_title(title_str, fontsize=18)
    return fig, ax


# Jacobian analysis
def _J11(u_in, u_net, f, tau_m, xt, af, tau_f):
    num = f*(2.*u_in - 3.*af*f - 2.*u_net)
    den = u_net*(-xt+u_net)
    ret = tau_m*xt/tau_f * num / den
    return ret


def _J12(u_in, u_net, f, tau_m, xt, af, tau_f):
    num = f**2.*(u_in*(xt-2.*u_net) + u_net**2. - af*f*(xt-2.*u_net))
    den = u_net**2.*(-xt+u_net)**2
    ret = tau_m*xt/tau_f * num / den
    return ret


def _J21(af, tau_f):
    return -af/tau_f


def _J22(tau_f):
    return -1./tau_f


def J_ralif(u_in, u_net, f, tau_m, xt, af, tau_f):
    ret = np.array([
        [_J11(u_in, u_net, f, tau_m, xt, af, tau_f),
         _J12(u_in, u_net, f, tau_m, xt, af, tau_f)],
        [_J21(af, tau_f), _J22(tau_f)]])
    return ret


def _f1(u_in, u_net, f, tau_m, xt, af, tau_f):
    num = f**2 * (u_in - af*f - u_net)
    den = u_net * (-xt + u_net)
    ret = tau_m*xt/tau_f*num/den
    return ret


def _f2(u_in, u_net, f, af, tau_f):
    ret = (u_in - af*f - u_net)/tau_f
    return ret


def rel_diff(x, x_ref):
    return (x-x_ref)/x_ref


def grad_check(fun, gradfun, x0, h=1e-5):
    yph = fun(x0+h)
    ymh = fun(x0-h)
    grad_num = (yph-ymh)/(2.*h)
    grad_th = gradfun(x0)
    return rel_diff(grad_num, grad_th)


def gradcheck_J(u_in, u_net, f, tau_m, xt, af, tau_f):
    f1_f = lambda x: _f1(u_in, u_net, x, tau_m, xt, af, tau_f)
    f1_u_net = lambda x: _f1(u_in, x, f, tau_m, xt, af, tau_f)
    f2_f = lambda x: _f2(u_in, u_net, x, af, tau_f)
    f2_u_net = lambda x: _f2(u_in, x, f, af, tau_f)

    J11_f = lambda x: _J11(u_in, u_net, x, tau_m, xt, af, tau_f)
    J12_u_net = lambda x: _J12(u_in, x, f, tau_m, xt, af, tau_f)
    J21_f = lambda x: _J21(af, tau_f)
    J22_u_net = lambda x: _J22(tau_f)

    print grad_check(f1_f, J11_f, f)
    print grad_check(f1_u_net, J12_u_net, u_net)
    print grad_check(f2_f, J21_f, f)
    print grad_check(f2_u_net, J22_u_net, u_net)


def plotcheck_J(tau_m, tref, xt, af, tau_f, dt, u_in=3.5):
    fig, axs = plt.subplots(
        nrows=4, ncols=2, gridspec_kw={'wspace': .3, 'hspace': .5},
        figsize=(14, 12))

    u_net = u_in
    x1, dx1 = np.linspace(5., .95/tref, 200, retstep=True)
    f1 = th_ralif_dfdt(u_net, u_in, x1, tau_m, tref, xt, af, tau_f)
    df1dx1 = _J11(u_in, u_net, x1, tau_m, xt, af, tau_f)
    df1dx1_num = np.diff(f1) / dx1

    axs[0, 0].plot(x1, f1)
    axs[1, 0].plot(x1, df1dx1, label='theory')
    axs[1, 0].plot(x1[1:], df1dx1_num, 'r', label='numerical')
    axs[1, 0].legend(loc='best')
    axs[1, 0].set_xlabel(r'$f$', fontsize=16)
    axs[0, 0].set_ylabel(r'$f_1$', fontsize=16)
    axs[1, 0].set_ylabel(r'$\frac{\partial f_1}{\partial{f}}$',
                         fontsize=20)

    f = num_ralif_fi(u_in, tau_m, tref, xt, af)
    x2, dx2 = np.linspace(1.1*xt, 5, 200, retstep=True)
    f1 = th_ralif_dfdt(x2, u_in, f, tau_m, tref, xt, af, tau_f)

    df1dx2 = _J12(u_in, x2, f, tau_m, xt, af, tau_f)
    df1dx2_num = np.diff(f1) / dx2

    axs[0, 1].plot(x2, f1)
    axs[1, 1].plot(x2, df1dx2, label='theory')
    axs[1, 1].plot(x2[1:], df1dx2_num, 'r', label='numerical')
    axs[1, 1].legend(loc='best')
    axs[1, 1].set_xlabel(r'$u_{net}$', fontsize=16)
    axs[0, 1].set_ylabel(r'$f_1$', fontsize=16)
    axs[1, 1].set_ylabel(r'$\frac{\partial f_1}{\partial{u_{net}}}$',
                         fontsize=20)

    f2 = _f2(u_in, u_net, x1, af, tau_f)
    df2dx1 = np.ones(len(x1)) * _J21(af, tau_f)
    df2dx1_num = np.diff(f2) / dx1

    axs[2, 0].plot(x1, f2)
    axs[3, 0].plot(x1, df2dx1, label='theory')
    axs[3, 0].plot(x1[1:], df2dx1_num, 'r', label='numerical')
    axs[3, 0].set_ylim(-2*af/tau_f, 0)
    axs[3, 0].legend(loc='best')
    axs[3, 0].set_xlabel(r'$f$', fontsize=16)
    axs[2, 0].set_ylabel(r'$f_2$', fontsize=16)
    axs[3, 0].set_ylabel(r'$\frac{\partial f_2}{\partial{f}}$',
                         fontsize=20)

    f2 = _f2(u_in, x2, f, af, tau_f)
    df2dx2 = np.ones(len(x2)) * _J22(tau_f)
    df2dx2_num = np.diff(f2) / dx2

    axs[2, 1].plot(x2, f2)
    axs[3, 1].plot(x2, df2dx2, label='theory')
    axs[3, 1].plot(x2[1:], df2dx2_num, 'r', label='numerical')
    axs[3, 1].set_ylim(-2./tau_f, 0)
    axs[3, 1].legend(loc='best')
    axs[3, 1].set_xlabel(r'$u_{net}$', fontsize=16)
    axs[2, 1].set_ylabel(r'$f_2$', fontsize=16)
    axs[3, 1].set_ylabel(r'$\frac{\partial f_2}{\partial{u_{net}}}$',
                         fontsize=20)


def contraction_analysis(u_in, tau_m, tref, xt, af, tau_f, dt=1e-4):
    T = 3.*tau_f
    u0 = u_in
    f0 = th_lif_fi(u0, tau_m, tref, xt)
    f, u_net = u_in_traj(u_in, tau_m, tref, xt, af, tau_f, dt, T, u0, f0)
    n = len(f)
    t = np.arange(n)*dt

    f_ss = num_ralif_fi(u_in, tau_m, tref, xt, af)
    u_net_ss = th_ralif_if(f_ss, tau_m, tref, xt, af)-af*f_ss

    delta = np.zeros((n, 2))
    delta[:, 0] = f_ss-f
    delta[:, 1] = u_net_ss-u_net
    dist_sq = np.zeros(n)
    ddist_sq = np.zeros(n)
    ddist_sq_lb = np.zeros(n)
    ddist_sq_ub = np.zeros(n)
    evals = np.zeros((n, 2))
    proj = np.zeros((n, 2))
    evects = (np.zeros((n, 2)), np.zeros((n, 2)))
    for i in xrange(n):
        J = J_ralif(u_in, u_net[i], f[i], tau_m, xt, af, tau_f)
        J_sym = .5*(J+J.T)
        lam, v = np.linalg.eigh(J_sym)
        evals[i, :] = lam
        evects[0][i, :] = v[:, 0]
        evects[1][i, :] = v[:, 1]
        dist_sq[i] = delta[i].dot(delta[i])
        ddist_sq[i] = 2.*delta[i].dot(J).dot(delta[i])
        ddist_sq_lb[i] = 2.*lam[0]*delta[i].dot(delta[i])
        ddist_sq_ub[i] = 2.*lam[1]*delta[i].dot(delta[i])
        proj[i, :] = v.T.dot(delta[i]).T

        # gradcheck_J(u_in, u_net[i], f[i], tau_m, xt, af, tau_f)
    ddist_sq_num = np.diff(dist_sq)/dt

    # switch up eigenvalues and vectors after crossover
    max_idx = np.argmax(evals[:, 0])
    evals[max_idx:, :] = evals[max_idx:, ::-1]
    proj[max_idx:, :] = proj[max_idx:, ::-1]
    __tmp0__ = evects[0][max_idx:, :].copy()
    __tmp1__ = evects[1][max_idx:, :].copy()
    evects[0][max_idx:, :] = __tmp1__
    evects[1][max_idx:, :] = __tmp0__

    retval = dict(
        t=t, u_net=u_net, f=f, delta=delta,
        dist_sq=dist_sq, ddist_sq=ddist_sq, ddist_sq_num=ddist_sq_num,
        ddist_sq_lb=ddist_sq_lb, ddist_sq_ub=ddist_sq_ub,
        eigenvals=evals, projections=proj, evects=evects)
    return retval


def plot_traj(t, u_net, f, delta, tau_m, tref, xt):
    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(121)
    u_lif = np.sort(np.linspace(0, 5, 50).tolist() + [xt, 1.01*xt])
    f_lif = th_lif_fi(u_lif, tau_m, tref, xt)
    ax.plot(u_lif, f_lif, 'k', label='open loop')
    ax.plot(u_net, f, 'mo', label=r'$f(u_{net}(t))$')
    ax.legend(loc='best')
    ax.set_xlabel(r'$u_{net}$', fontsize=16)
    ax.set_ylabel(r'$f$', fontsize=16)
    ax = fig.add_subplot(222)
    ax.plot(t, delta[:, 0])
    ax.set_ylabel(r'$\delta x_1$', fontsize=16)
    ax = fig.add_subplot(224)
    ax.plot(t, delta[:, 1])
    ax.set_xlabel(r'$t$', fontsize=16)
    ax.set_ylabel(r'$\delta x_2$', fontsize=16)


def plot_contraction(t, dist_sq,
                     ddist_sq_num, ddist_sq, ddist_sq_lb, ddist_sq_ub,
                     eigenvals, projections):
    # plot deltax deltax
    fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True, figsize=(15, 8))
    fig.subplots_adjust(hspace=.3)
    axs[0, 0].plot(t, dist_sq)
    axs[1, 0].plot(t[1:], ddist_sq_num, 'r', label='observed')
    axs[1, 0].plot(t, ddist_sq, 'g', lw=2, label='theory')
    axs[1, 0].plot(t, ddist_sq_lb, 'b', label='theory lower bound')
    axs[1, 0].plot(t, ddist_sq_ub, 'k', label='theory upper bound')
    axs[1, 0].axhline(0, c='k', alpha=.2)
    ylims = 1.1*np.max([max(np.abs(ddist_sq_num)), max(np.abs(ddist_sq))])
    ylims = (-ylims, ylims)
    axs[1, 0].set_ylim(ylims)
    axs[1, 0].legend(loc='lower right')
    axs[1, 0].set_xlabel(r'$t$', fontsize=16)
    axs[1, 1].set_xlabel(r'$t$', fontsize=16)
    axs[0, 0].set_title(r'$\delta\mathbf{x}^T\delta\mathbf{x}$', fontsize=16)
    axs[1, 0].set_title(r'$\frac{d}{dt}(\delta\mathbf{x}^T\delta\mathbf{x})$',
                        fontsize=16, y=1.03)

    # plot eigenvalue analysis
    axs[0, 1].plot(t, eigenvals)
    axs[1, 1].plot(t, projections)
    axs[0, 1].axhline(0, c='k')
    axs[1, 1].axhline(0, c='k')
    axs[0, 1].set_title(
        'eigenvalues of ' + r'$\partial\mathbf{f}/\partial\mathbf{x}$',
        fontsize=16)
    axs[1, 1].set_title((
        r'$\delta\mathbf{x}$' + ' projected on to eigenvectors of ' +
        r'$\partial\mathbf{f}/\partial\mathbf{x}$'), fontsize=16)


def plot_components(t, eigenvals, projections, evects):
    # plot components of ddistsq
    ddist_sq_components = 2 * projections**2 * eigenvals
    fig = plt.figure(figsize=(15, 6))
    ax = fig.add_subplot(121)
    label1str = (r'$2\lambda_1\delta\mathbf{x}^T\mathbf{v}_1$' +
                 r'$\mathbf{v}_1^T\delta\mathbf{x}$')
    ax.plot(t, ddist_sq_components[:, 0], label=label1str)
    label2str = (r'$2\lambda_2\delta\mathbf{x}^T\mathbf{v}_2$' +
                 r'$\mathbf{v}_2^T\delta\mathbf{x}$')
    ax.plot(t, ddist_sq_components[:, 1], label=label2str)
    labelstr = label1str + r' $+$ ' + label2str
    ax.plot(t, np.sum(ddist_sq_components, axis=1), label=labelstr)
    ax.axhline(0, c='k', alpha=.2)
    ax.legend(loc='upper right')
    ax.set_xlabel(r'$t$', fontsize=16)
    titlestr = (
        'components of ' +
        r'$2\delta\mathbf{x}$' +
        r'$\frac{\partial\mathbf{f}}{\partial\mathbf{x}}$' +
        r'$\delta\mathbf{x}$')
    ax.set_title(titlestr, fontsize=16, y=1.025)
    ax = fig.add_subplot(222)
    ax.plot(t, evects[0][:, ::-1])
    ax.set_ylabel(r'$\mathbf{v}_1$', fontsize=16)
    ax.set_title('eigenvector components', fontsize=16)
    ax = fig.add_subplot(224)
    ax.plot(t, evects[1][:, ::-1])
    ax.set_xlabel(r'$t$', fontsize=16)
    ax.set_ylabel('$\mathbf{v}_2$', fontsize=16)

    # plot eigenvectors onto unit circle
    th = np.linspace(0, 2*np.pi, 50)
    x = np.cos(th)
    y = np.sin(th)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(x, y, c='k', alpha=.2)
    ax.plot(evects[0][:, 1], evects[0][:, 0], 'r-o')
    ax.plot(evects[1][:, 1], evects[1][:, 0], 'b-o')
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_xlabel(r'$x_1$', fontsize=16)
    ax.set_ylabel(r'$x_2$', fontsize=16)
    ax.set_title(
        'eigenvectors of ' + r'$\partial\mathbf{f}/\partial\mathbf{x}$',
        fontsize=16)
