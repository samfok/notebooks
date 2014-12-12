# utility functions for LIF_neuron.ipynb
import numpy as np
import multiprocessing
from signal import make_poisson_spikes, filter_spikes
from neuron import th_lif_fi, run_lifsoma
from plot import (
    plot_continuous, plot_spike_raster, plot_histogram,
    plot_contour, plot_scatter, match_xlims, save_close_fig)
from scipy.special import erf
import matplotlib.pyplot as plt


def normal_cdf(x, mu=0., sigma=1.):
    return 1./2.*(1+erf((x-mu)/(sigma*np.sqrt(2))))


# Gaussian 1 sigma percentiles
s1_percentiles = [normal_cdf(-1.)*100., normal_cdf(1.)*100.]


rng = None
FIGDIR = None


def set_utils(r, figdir):
    global rng, FIGDIR
    rng = r
    FIGDIR = figdir


def _set_io_collector(*args):
    global io_collector
    io_collector = args[0]


def _call_compute_stats(fexc_finh):
    """for using multiprocessing with a class method"""
    return io_collector._compute_stats(fexc_finh)


class LIF_IO_Collector(object):
    """Collects output stats from an LIF neuron given Poisson spike input"""
    def __init__(self, dt, T, alpha, neuronp, filt_tau, k_trans,
                 dev_pcts=s1_percentiles):
        self.dt = dt
        self.T = T
        self.alpha = alpha
        self.neuronp = neuronp
        self.filt_tau = filt_tau
        self.k_trans = k_trans
        self.nsteps = int(np.ceil(T/dt))
        self.dev_pcts = dev_pcts

    def _compute_stats(self, fexc_finh):
        fexc = fexc_finh[0]
        finh = fexc_finh[1]

        fexc_nspikes = 2.*self.T*fexc
        finh_nspikes = 2.*self.T*finh
        fexc_spks_in = make_poisson_spikes(fexc, fexc_nspikes, rng)
        finh_spks_in = make_poisson_spikes(finh, finh_nspikes, rng)
        t, fexc_in = filter_spikes(self.dt, self.T, fexc_spks_in,
                                   self.neuronp['tau_syn'])
        t, finh_in = filter_spikes(self.dt, self.T, finh_spks_in,
                                   self.neuronp['tau_syn'])

        u_in = self.alpha*(fexc_in - finh_in)
        spk_t = run_lifsoma(self.dt, u_in, self.neuronp['tau_m'],
                            self.neuronp['tref'], self.neuronp['xt'])
        t, rate_out = filter_spikes(self.dt, self.T, spk_t, self.filt_tau)

        ss_idx = t > self.k_trans*self.neuronp['tau_syn']
        tuning = np.mean(rate_out[ss_idx])
        pct_obs = np.percentile(rate_out[ss_idx], self.dev_pcts)
        dev1s_l = tuning-pct_obs[0]
        dev1s_u = pct_obs[1]-tuning

        return [tuning, dev1s_l, dev1s_u]

    def collect_io_stats(self, fexc, finh=None, ret_devs=False, max_proc=1):
        shape = fexc.shape
        fexc = fexc.flatten()
        if finh is not None:
            finh = finh.flatten()
            assert fexc.shape == finh.shape
        else:
            finh = np.zeros(fexc.shape)

        tuning = np.zeros(fexc.shape)
        dev1s_l = np.zeros(fexc.shape)
        dev1s_u = np.zeros(fexc.shape)

        fexc.shape = -1, 1
        finh.shape = -1, 1
        fexc_finh = np.hstack((fexc, finh))

        n_proc = min(fexc.shape[0], multiprocessing.cpu_count()-1)
        if max_proc is not None:
            n_proc = min(n_proc, max_proc)

        if n_proc == 1:
            results = map(self._compute_stats, fexc_finh)
        else:
            worker_pool = multiprocessing.Pool(
                processes=n_proc, initializer=_set_io_collector,
                initargs=(self,))
            results = worker_pool.map(_call_compute_stats, fexc_finh)
            worker_pool.close()
            worker_pool.join()

        for idx, result in enumerate(results):
            tuning[idx] = result[0]
            dev1s_l[idx] = result[1]
            dev1s_u[idx] = result[2]

        tuning = tuning.reshape(shape)
        dev1s_l = dev1s_l.reshape(shape)
        dev1s_u = dev1s_u.reshape(shape)

        if ret_devs:
            retval = tuning, dev1s_l, dev1s_u
        else:
            retval = tuning
        return retval


# define a function for sweeping the input space of
# an LIF neuron and measuring its output statistics

def fexc_finh_sweep(neuronp, filt_tau=.01, k_trans=5, max_u=6., max_f=1000.,
                    npts=10, fname_pre='', max_proc=None, close=False):
    alpha = max_u/max_f

    fexc = np.linspace(0, max_f, npts)
    finh = np.linspace(0, max_f, npts)
    fexc_g, finh_g = np.meshgrid(fexc, finh)
    lam_g = fexc_g+finh_g
    u_g = alpha*(fexc_g-finh_g)
    s_g = np.sqrt(lam_g/(2.*neuronp['tau_syn']))
    tuning_th = th_lif_fi(u_g,
                          neuronp['tau_m'], neuronp['tref'], neuronp['xt'])
    T = 2.
    dt = .0001

    io_collector = LIF_IO_Collector(
        dt=dt, T=T, alpha=alpha, neuronp=neuronp, filt_tau=filt_tau,
        k_trans=k_trans)
    tuning, dev1s_l, dev1s_u = io_collector.collect_io_stats(
        fexc=fexc_g, finh=finh_g, ret_devs=True, max_proc=max_proc)

    # E[u]
    fig, ax = plot_contour(
        fexc_g, finh_g, u_g,
        contourfp={'cmap': plt.cm.BrBG}, contourp={'colors': 'r'},
        figp={'figsize': (8, 6)},
        xlabel=r'$f_{exc}$', xlabelp={'fontsize': 20},
        ylabel=r'$f_{inh}$', ylabelp={'fontsize': 20},
        title=r'$E[u]$', titlep={'fontsize': 20})
    plot_scatter(
        fexc_g, finh_g, scatterp={'c': 'm', 'marker': '+', 'alpha': .5}, ax=ax,
        xlim=(fexc[0], fexc[-1]), ylim=(finh[0], finh[-1]), close=close,
        fname=FIGDIR+fname_pre+'fe_fi_u.png', savep={'dpi': 200})

    # a(E[u]))
    fig, ax = plot_contour(
        fexc_g, finh_g, tuning_th,
        contourfp={'cmap': plt.cm.copper}, contourp={'colors': 'r'},
        figp={'figsize': (8, 6)},
        xlabel=r'$f_{exc}$', xlabelp={'fontsize': 20},
        ylabel=r'$f_{inh}$', ylabelp={'fontsize': 20},
        title=r'$a(E[u])$', titlep={'fontsize': 20})
    plot_scatter(
        fexc_g, finh_g, scatterp={'c': 'm', 'marker': '+', 'alpha': .5}, ax=ax,
        xlim=(fexc[0], fexc[-1]), ylim=(finh[0], finh[-1]), close=close,
        fname=FIGDIR+fname_pre+'fe_fi_tuning_th.png', savep={'dpi': 200})

    # sqrt(Var(u))
    fig, ax = plot_contour(
        fexc_g, finh_g, s_g,
        contourfp={'cmap': plt.cm.PuOr}, contourp={'colors': 'r'},
        figp={'figsize': (8, 6)},
        xlabel=r'$f_{exc}$', xlabelp={'fontsize': 20},
        ylabel=r'$f_{inh}$', ylabelp={'fontsize': 20},
        title=r'$\sigma(f_{exc}+f_{inh})$', titlep={'fontsize': 20})
    plot_scatter(fexc_g, finh_g,
                 scatterp={'c': 'm', 'marker': '+', 'alpha': .5}, ax=ax,
                 xlim=(fexc[0], fexc[-1]), ylim=(finh[0], finh[-1]),
                 close=close,
                 fname=FIGDIR+fname_pre+'fe_fi_noise.png', savep={'dpi': 200})

    # E[a(u)]
    fig, ax = plot_contour(
        fexc_g, finh_g, tuning,
        contourfp={'cmap': plt.cm.copper}, contourp={'colors': 'r'},
        figp={'figsize': (8, 6)},
        xlabel=r'$f_{exc}$', xlabelp={'fontsize': 20},
        ylabel=r'$f_{inh}$', ylabelp={'fontsize': 20},
        title=r'$E[a(u)]$', titlep={'fontsize': 20})
    plot_scatter(
        fexc_g, finh_g, scatterp={'c': 'm', 'marker': '+', 'alpha': .5}, ax=ax,
        xlim=(fexc[0], fexc[-1]), ylim=(finh[0], finh[-1]), close=close,
        fname=FIGDIR+fname_pre+'fe_fi_noisy_tuning.png', savep={'dpi': 200})

    # Var(a(u))
    fig, ax = plot_contour(
        fexc_g, finh_g, dev1s_l,
        contourfp={'cmap': plt.cm.winter}, contourp={'colors': 'r'},
        subplotp=(1, 2, 1), figp={'figsize': (16, 6)},
        xlabel=r'$f_{exc}$', xlabelp={'fontsize': 20},
        ylabel=r'$f_{inh}$', ylabelp={'fontsize': 20},
        title=r'$-\sigma\%(a(u))$', titlep={'fontsize': 20})
    plot_scatter(
        fexc_g, finh_g, scatterp={'c': 'm', 'marker': '+', 'alpha': .5}, ax=ax,
        xlim=(fexc[0], fexc[-1]), ylim=(finh[0], finh[-1]))
    fig, ax = plot_contour(
        fexc_g, finh_g, dev1s_u,
        contourfp={'cmap': plt.cm.winter}, contourp={'colors': 'r'}, fig=fig,
        subplotp=(1, 2, 2),
        xlabel=r'$f_{exc}$', xlabelp={'fontsize': 20},
        ylabel=r'$f_{inh}$', ylabelp={'fontsize': 20},
        title=r'$+\sigma\%(a(u))$', titlep={'fontsize': 20})
    plot_scatter(
        fexc_g, finh_g, scatterp={'c': 'm', 'marker': '+', 'alpha': .5}, ax=ax,
        xlim=(fexc[0], fexc[-1]), ylim=(finh[0], finh[-1]), close=close,
        fname=FIGDIR+fname_pre+'fe_fi_noisy_tuning.png', savep={'dpi': 200})


# Define a class for collecting the synapse, soma, and filtered output
# statistics of an LIF neuron given Poisson spiking input
def _worker_init(dt_init, alpha_init, xlim_T_init, fname_init, close_init,
                 tgt_outspikes_init, neuronp_init):
    # these globals only contained within each worker
    global dt, alpha, xlim_T, fname, close, tgt_outspikes
    global tau_syn, tau_m, tref, xt
    dt = dt_init
    alpha = alpha_init
    xlim_T = xlim_T_init
    fname = fname_init
    close = close_init
    tgt_outspikes = tgt_outspikes_init
    tau_syn = neuronp_init['tau_syn']
    tau_m = neuronp_init['tau_m']
    tref = neuronp_init['tref']
    xt = neuronp_init['xt']


def _worker_lif_stats(uf):
    u = uf[0]
    rate = uf[1]

    fig = plt.figure(figsize=(16, 6))
    ax = [fig.add_subplot(4, 2, 1)]
    for i in xrange(1, 4):
        ax.append(fig.add_subplot(4, 2, i*2+1))

    th_f = th_lif_fi(u, tau_m, tref, xt)
    T = 1.
    if th_f > 0:
        T = max(T, tgt_outspikes/th_f)

    nspikes = max(10, 2.*T*rate)
    spks_in = make_poisson_spikes(rate, nspikes, rng)
    plot_spike_raster(spks_in, ax=ax[0], yticks=[])

    t, u_in = filter_spikes(dt, T, spks_in, tau_syn)
    u_in *= alpha
    mean_uin = np.mean(u_in[t > 5*tau_syn])
    plot_continuous(t, u_in, ax=ax[1],
                    axhline=mean_uin, axhlinep={'color': 'r'},
                    ylabel='syn', ylabelp={'fontsize': 16})

    spk_t, state = run_lifsoma(dt, u_in, tau_m, tref, xt, ret_state=True)
    plot_continuous(t, state, ax=ax[2],
                    ylabel='soma', ylabelp={'fontsize': 16})

    t, rate_out = filter_spikes(dt, xlim_T, spk_t, tau_syn)
    ss_idx = t > 5*tau_syn
    mean_rate_out = np.mean(rate_out[ss_idx])
    var_rate_out = np.var(rate_out[ss_idx])
    plot_continuous(t, rate_out, ax=ax[3],
                    axhline=mean_rate_out, axhlinep={'color': 'r'},
                    xlabel=r'$t$ (s)', xlabelp={'fontsize': 20},
                    ylabel=r'filtered out', ylabelp={'fontsize': 16})
    ax[3].axhline(th_f, color='r', linestyle=':')

    for a in ax[:-1]:
        plt.setp(a.get_xticklabels(), visible=False)
    match_xlims(ax, (0, xlim_T))
    ax[0].set_title(r'$\alpha=%.1e$,  $E[u]=%.2f$,  $f_{in}=%.1f$ Hz' %
                    (alpha, u, rate), fontsize=20)

    if len(spk_t) > 5:
        ax = fig.add_subplot(1, 2, 2)
        isi = np.diff(spk_t)
        histp = dict(bins=int(tgt_outspikes/10), normed=True, histtype='step')
        plot_histogram(
            isi, histp=histp, ax=ax,
            axvline=1./mean_rate_out,
            axvlinep={'color': 'r', 'label': 'observed mean'},
            xlabel='output isi (s)', xlabelp={'fontsize': 20},
            xlim=(min(isi), max(isi)),
            title=r'$%d$ spikes, $E[a(u)]=%.1f$, $Var(a(u))=%.1f$' %
            (len(spk_t), mean_rate_out, var_rate_out),
            titlep={'fontsize': 20})
        if th_f > 0:
            ax.axvline(1./th_f, color='r', linestyle=':',
                       label='theoretical mean')
        ax.legend(loc='upper right')
    sub_fname = fname
    if fname is not None:
        sub_fname = fname + '_alpha%.1e_u%.2f_f%.1f_.png' % (alpha, u, rate)
    save_close_fig(fig, sub_fname, close)


def lif_stats(u, dt=.0001, max_f=1000., xlim_T=1.,
              fname=None, close=False, tgt_outspikes=1000, max_processes=None):
    neuronp = dict(tau_syn=.01, tau_m=.01, tref=.005, xt=1.)

    max_u = max(u)

    alpha = max_u/max_f
    f = u/alpha

    n_processes = min(len(u), multiprocessing.cpu_count()-1)
    if max_processes is not None:
        n_processes = min(n_processes, max_processes)

    worker_init_args = (
        dt, alpha, xlim_T, fname, close, tgt_outspikes, neuronp)

    if n_processes == 1:
        _worker_init(*worker_init_args)
        map(_worker_lif_outstats, zip(u, f))
    else:
        worker_pool = multiprocessing.Pool(
            processes=n_processes, initializer=_worker_init,
            initargs=worker_init_args)
        worker_pool.map(_worker_lif_stats, zip(u, f))
        worker_pool.close()
        worker_pool.join()
