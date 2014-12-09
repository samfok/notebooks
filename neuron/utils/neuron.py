# define neuron models
import numpy as np


def th_lif_fi(u, tau, tref, xt):
    """Theoretical LIF tuning curve

    Calculates firing rate from input with
    f = (tref - tau*ln(1-xt/u))**-1, u >  xt
        0                          , u <= xt

    Parameters
    ----------
    u : array-like of floats
        input
    tau : float
        membrane time constant
    tref : float
        refractory period
    xt : float
        threshold
    """
    # handle scalars
    if isinstance(u, (int, float)):
        u = np.array([u])
    # handle arrays
    f = np.zeros(u.shape)
    idx = u > xt
    f[idx] = (tref-tau*np.log((u[idx]-xt)/(u[idx])))**(-1.)
    return f


def th_lif_if(f, tau, tref, xt):
    """Theoretical inverse of the LIF tuning curve

    Computes input from firing rate with
    u = xt/(1-exp((tref-1/f)/tau))
    Parameters
    ----------
    f : array-like of floats
        firing rates (must be > 0)
    tau : float
        membrane time constant
    tref : float
        refractory period
    xt : float
        threshold
    """
    assert (f > 0.).all(), "LIF tuning curve only invertible for f>0."
    u = xt/(1-np.exp((tref-1./f)/tau))
    return u


def iter_alif_fi(u, tau, tref, xt, af=1e-3, max_iter=100, rel_tol=1e-3,
                 verbose=False):
    """Iteratively find rate-based adaptive LIF tuning curve

    Uses binary search to find the steady state firing rate of an adaptive LIF
    neuron.

    Parameters
    ----------
    u : array-like of floats
        input
    tau : float
        membrane time constant
    tref : float
        refractory period
    xt : float
        threshold
    af : float (optional)
        scales the inhibitory feedback
    max_iter : int (optional)
        maximium number of iterations in binary search
    rel_tol : float (optional)
        relative tolerance of binary search algorithm. algorithm terminates
        when maximum difference between estimated af and input af is within
        tol*af
    """
    assert af > 0, "inhibitory feedback scaling must be > 0"
    # handle scalars
    if isinstance(u, (int, float)):
        u = np.array([u])

    f_high = th_lif_fi(u, tau, tref, xt)
    f_ret = np.zeros(f_high.shape)
    idx = f_high > 0.
    f_high = f_high[idx]
    f_low = np.zeros(f_high.shape)
    tol = abs(af)*rel_tol  # set tolerance relative to af
    while True:
        f = (f_high+f_low)/2.
        u_net = th_lif_if(f, tau, tref, xt)
        u_f = u[idx] - u_net
        a = u_f/f
        high_idx = a > af
        low_idx = a <= af
        f_high[low_idx] = f[low_idx]
        f_low[high_idx] = f[high_idx]

        max_diff = np.max(np.abs(a-af))
        if max_diff < tol:
            exit_msg = 'reached tolerance'
            break
        max_iter -= 1
        if max_iter == 0:
            exit_msg = 'reached max iterations'
            break
    f_ret[idx] = f
    if verbose:
        print exit_msg
    return f_ret
