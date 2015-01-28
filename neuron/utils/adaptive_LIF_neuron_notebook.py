import numpy as np
from matplotlib.pyplot import figure
from multiprocessing import Pool, cpu_count
from neuron import (
    th_lif_fi, th_lif_if, num_alif_fi, num_alif_fi_mu_apx, sim_alif_fi,
    run_alifsoma)
from signal import filter_spikes


def sim_vs_num_tauf(dt, T, max_u, taum, tref, xt, af, tauf, ax=None):
    n = len(tauf)
    if ax is None:
        fig = figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
    cc = [(0, 0, .5*float(i)/n+.5) for i in xrange(n)]

    th_max_f = 1./tref
    u = np.array(np.sort(np.linspace(0, max_u, 100).tolist() +
                         [xt, 1.001*xt, 1.01*xt, 1.1*xt]))
    lif_fi = th_lif_fi(u, taum, tref, xt)
    max_f = max(lif_fi)
    f = np.linspace(5., max_f, 10)
    _u = th_lif_if(f, taum, tref, xt)  # linear in firing rate

    ax.plot(u, lif_fi, 'k', lw=2, label='nonadaptive LIF')
    for idx, tauf_val in enumerate(tauf):
        sim_af = sim_alif_fi(dt, _u, taum, tref, xt, af, tauf_val)
        num_af = num_alif_fi(u, taum, tref, xt, af, tauf_val, max_f=max_f)
        ax.plot(_u, sim_af, 'o', mfc=cc[idx], ms=8, alpha=.5)
        ax.plot(u, num_af, c=cc[idx], lw=2,
                label=r'$\tau_f=%.3f$' % (tauf_val))
    ax.set_ylim(0, th_max_f)
    ax.set_xlim(0, max_u*1.001)
    ax.legend(loc='upper left')
    ax.set_xlabel(r'$u_{in}$', fontsize=20)
    ax.set_ylabel(r'$f$ (spks / s)', fontsize=20)
    ax.set_title(r'$\tau=%.3f$, $\alpha_f=%.3f$' % (taum, af), fontsize=20)


def sim_vs_num_af(dt, T, max_u, taum, tref, xt, af, tauf, ax=None):
    n = len(af)
    if ax is None:
        fig = figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
    cc = [(0, 0, .5*float(i)/n+.5) for i in xrange(n)]

    th_max_f = 1./tref
    u = np.array(np.sort(np.linspace(0, max_u, 100).tolist() +
                         [xt, 1.001*xt, 1.01*xt, 1.1*xt]))
    lif_fi = th_lif_fi(u, taum, tref, xt)
    max_f = max(lif_fi)
    f = np.linspace(5., max_f, 10)
    _u = th_lif_if(f, taum, tref, xt)  # linear in firing rate

    ax.plot(u, lif_fi, 'k', lw=2, label='nonadaptive LIF')
    for idx, af_val in enumerate(af):
        sim_af = sim_alif_fi(dt, _u, taum, tref, xt, af_val, tauf)
        num_af = num_alif_fi(u, taum, tref, xt, af_val, tauf, max_f=max_f)
        ax.plot(u, num_af, c=cc[idx], lw=2,
                label=r'$\alpha_f=%.3f$' % (af_val))
        ax.plot(_u, sim_af, 'o', mfc=cc[idx], ms=8, alpha=.5)
    ax.set_ylim(0, th_max_f)
    ax.set_xlim(0, max_u*1.001)
    ax.legend(loc='upper left')
    ax.set_xlabel(r'$u_{in}$', fontsize=20)
    ax.set_ylabel(r'$f$ (spks / s)', fontsize=20)
    ax.set_title(r'$\tau=%.3f$, $\tau_f=%.3f$' % (taum, tauf), fontsize=20)


def _af_tauf_sweep_worker(args):
    return _af_tauf_sweep_worker_unwrapped(*args)


def _af_tauf_sweep_worker_unwrapped(dt, uin, taum, tref, xt, af, tauf, tausyn):
    if tausyn is None:
        tausyn = tauf
    T = max([4.*tauf, 4.*tausyn,
             2.0/num_alif_fi(uin, taum, tref, xt, af, tauf)])
    nsteps = int(np.ceil(T/dt))
    t = np.arange(nsteps)*dt
    u = np.zeros(nsteps)+uin
    f_init = th_lif_fi(uin, taum, 0., xt)
    isi_init = 1./f_init
    f_ss = num_alif_fi(uin, taum, tref, xt, af, tauf)
    isi_ss = 1./f_ss
    n_ss_spks = int(np.ceil(T/isi_ss))
    f_ss_spk_times = isi_ss*np.arange(n_ss_spks)+isi_init
    x_syn = filter_spikes(dt, T, f_ss_spk_times, tausyn, ret_time=False)
    e_x_syn = np.zeros(t.shape)
    idx = t > isi_init
    e_x_syn[idx] = f_ss*(1-np.exp(-(t[idx]-isi_init)/tausyn))
    alif_spk_times = run_alifsoma(dt, u, taum, tref, xt, af=af, tauf=tauf)
    ax_syn = filter_spikes(dt, T, alif_spk_times, tausyn, ret_time=False)
    ret = dict(t=t, T=T, af=af, tauf=tauf, tausyn=tausyn, f_ss=f_ss,
               ax_syn=ax_syn, x_syn=x_syn, e_x_syn=e_x_syn)
    return ret


def af_tauf_sweep(dt, uin, taum, tref, xt, afs, taufs, tausyn=None,
                  max_proc=cpu_count()-1, suptitle=None):
    n_af = len(afs)
    n_tauf = len(taufs)
    params = [(dt, uin, taum, tref, xt, af, tauf, tausyn)
              for tauf in taufs for af in afs]

    fig = figure(figsize=(16, 12))
    axs = [fig.add_subplot(n_tauf, n_af, i+1) for i in xrange(n_af*n_tauf)]
    if (max_proc in (None, 0, 1)) or (len(params) == 1):
        results = map(_af_tauf_sweep_worker, params)
    else:
        workers = Pool(max_proc)
        results = workers.map(_af_tauf_sweep_worker, params)
        workers.close()
        workers.join()

    for idx, r in enumerate(results):
            ax = axs[idx]
            ax.plot(r['t']/r['tausyn'], r['ax_syn']/r['f_ss'], 'b',
                    label='adaptive')
            ax.plot(r['t']/r['tausyn'], r['x_syn']/r['f_ss'], 'r', alpha=.3,
                    label='nonadaptive')
            ax.plot(r['t']/r['tausyn'], r['e_x_syn']/r['f_ss'], 'r:', alpha=.7,
                    label='E[nonadaptive]')
            ax.axhline(1., c='k', ls=':')
            ax.set_xlim(0, r['T']/r['tausyn'])
            ax.set_title(r'$\alpha_f,\tau_f=%.3f,\ %.3f$' %
                         (r['af'], r['tauf']), fontsize=14)
    axs[12].legend(loc='lower right')
    fig.text(.48, .08, r'$t (\tau_{syn})$', fontsize=20)
    fig.text(.08, .52, r'$x_{syn}/\lambda_{ss}$', fontsize=20,
             rotation='vertical')
    if suptitle is not None:
        fig.text(.48, .93, suptitle, fontsize=20)


def taylor1_lif_k0_k1(a, tau, tref, xt):
    k1 = tau*xt/((tref-tau*np.log(1-xt/a))**2*a*(a-xt))
    k0 = th_lif_fi(a, tau, tref, xt) - k1*a
    return k0, k1


def taylor1_alif_fi(u, tau_m, tref, xt, af, tau_f):
    af *= np.exp(-tref/tau_f)
    f = np.zeros(u.shape)
    idx = u > xt
    k0, k1 = taylor1_lif_k0_k1(u[idx], tau_m, tref, xt)
    f[idx] = (k0+k1*u[idx])/(1+k1*af)
    return f


def compare_mu_apx(max_u, tau_m, tref, xt, af, tau_f):
    th_max_f = 1./tref
    # inputs
    u = np.array(np.sort(np.linspace(0, max_u, 100).tolist() + [xt]))
    max_f = th_lif_fi(max_u, tau_m, tref, xt)
    f = np.linspace(5., max_f, 13)
    u_pts = th_lif_if(f, tau_m, tref, xt)  # linear in firing rate

    # fi curves
    lif_fi = th_lif_fi(u, tau_m, tref, xt)
    alif_fi = num_alif_fi(u, tau_m, tref, xt, af, tau_f)
    alif_fi_mu_apx = num_alif_fi_mu_apx(u, tau_m, tref, xt, af, tau_f)
    alif_fi_mu_apx_taylor1 = taylor1_alif_fi(u, tau_m, tref, xt, af, tau_f)
    pts_alif_fi = num_alif_fi(u_pts, tau_m, tref, xt, af, tau_f)
    pts_alif_fi_mu_apx = num_alif_fi_mu_apx(u_pts, tau_m, tref, xt, af, tau_f)
    pts_alif_fi_mu_apx_taylor1 = taylor1_alif_fi(u_pts, tau_m, tref, xt, af,
                                                 tau_f)

    # plot fi curves
    alpha = .5
    fig = figure(figsize=(16, 6))
    ax = fig.add_subplot(121)
    ax.plot(u, lif_fi, 'k', linewidth=2, label='LIF')
    ax.plot(u, alif_fi, 'b', linewidth=2, label='aLIF')
    ax.plot(u, alif_fi_mu_apx, 'c', linewidth=2, label='aLIF mean feedback')
    ax.plot(u, alif_fi_mu_apx_taylor1, 'm', linewidth=2,
            label='aLIF mean feedback linear apx')
    ax.plot(u_pts, pts_alif_fi, 'bo', ms=7, alpha=alpha)
    ax.plot(u_pts, pts_alif_fi_mu_apx, 'co', ms=7, alpha=alpha)
    ax.plot(u_pts, pts_alif_fi_mu_apx_taylor1, 'mo', ms=7, alpha=alpha)
    ax.set_ylim(0, th_max_f)
    ax.set_xlim(0, max_u*1.001)
    ax.legend(loc='upper left')
    ax.set_xlabel(r'$u_{in}$', fontsize=20)
    ax.set_ylabel(r'$f(u_{in})$ (spks / s)', fontsize=20)

    # fg linear approximation
    uf_f = np.linspace(0, th_max_f, 201)
    uf = af*np.exp(-tref/tau_f)*uf_f
    k0, k1 = taylor1_lif_k0_k1(u_pts, tau_m, tref, xt)
    taylor1_f_uf = np.zeros((u_pts.shape[0], uf.shape[0]))
    for idx, u_val in enumerate(u_pts):
        taylor1_f_uf[idx, :] = k0[idx]+k1[idx]*u_val-k1[idx]*uf

    uf_alif = af*np.exp(-tref/tau_f)*pts_alif_fi
    uf_alif_mu_apx = af*np.exp(-tref/tau_f)*pts_alif_fi_mu_apx
    uf_alif_mu_apx_taylor1 = af*np.exp(-tref/tau_f)*pts_alif_fi_mu_apx_taylor1

    # plot fg curves
    cc = [(float(i)/len(u_pts), 0, 0) for i in xrange(len(u_pts))]
    ax = fig.add_subplot(122)
    for idx, u_val in enumerate(u_pts):
        net_u = u_val - uf
        f_ss = th_lif_fi(net_u, tau_m, tref, xt)
        line = ax.plot(uf, f_ss, color=cc[idx])[0]
        if u_val in (u_pts[0], u_pts[-1]):
            line.set_label(r'$u_{in}=%.2f$' % u_val)
        ax.plot(uf, taylor1_f_uf[idx, :], color=cc[idx], ls=':')
    ax.plot(uf, uf_f, 'b', label=r"$u_f=\alpha_f'f$")
    ax.plot(uf_alif, pts_alif_fi, 'bo', ms=8, alpha=alpha)
    ax.plot(uf_alif_mu_apx, pts_alif_fi_mu_apx, 'co', ms=8, alpha=alpha)
    ax.plot(uf_alif_mu_apx_taylor1, pts_alif_fi_mu_apx_taylor1, 'mo', ms=8,
            alpha=alpha)
    ax.set_ylim(0, th_max_f)
    ax.set_xlim(0, max(uf))
    ax.legend(loc='upper left', fontsize=16)
    ax.set_ylim(0, th_max_f)
    ax.set_xlabel(r'$E[u_f]$', fontsize=20)
    ax.set_ylabel(r'$f(u_f;\ u_{in})$ (spks / s)', fontsize=20)
    fig.suptitle(r'$\tau=%.3f$, $t_{ref}=%.3f$, $\alpha_f=%.3f$, $\tau_f=%.3f$'
                 % (tau_m, tref, af, tau_f), fontsize=20)
