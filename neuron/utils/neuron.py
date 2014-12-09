# define neuron models
import numpy as np


def th_lif_fi(u, tau, tref, xt):
    """Theoretical LIF tuning curve"""
    # handle scalars
    if isinstance(u, (int, float)):
        u = np.array([u])
    # handle arrays
    f = np.zeros(u.shape)
    idx = u > xt
    f[idx] = (tref-tau*np.log((u[idx]-xt)/(u[idx])))**(-1.)
    return f


def th_lif_if(f, tau, tref, xt):
    """Theoretical inverse of the LIF tuning curve"""
    assert (f > 0.).all(), "LIF tuning curve only invertible for f>0."
    u = xt/(1-np.exp((tref-1./f)/tau))
    return u


def iter_alif_fi(u, tau, tref, xt, af=-1e-3, max_iter=100, tol=1e-3,
                 verbose=False):
    """Iteratively find rate-based adaptive LIF tuning curve

    This is not a very robust method -- susceptible to numerical instability
    """
    f = th_lif_fi(u, tau, tref, xt)  # initialize at standard lif tuning value
    f_prev = f.copy()
    while True:
        uf = af*f
        net_u = u+uf
        f = th_lif_fi(net_u, tau, tref, xt)
        max_diff = np.max(np.abs(f_prev - f))
        if max_diff < tol:
            exit_msg = 'reached tolerance'
            break
        max_iter -= 1
        if max_iter == 0:
            exit_msg = 'reached max iterations'
            break
        clip_idx = np.logical_and(f == 0, f_prev > 0)  # idx of 0-ed entries
        f[clip_idx] = f_prev[clip_idx]*.9
        f_prev = f
    if verbose:
        print exit_msg
    return f
