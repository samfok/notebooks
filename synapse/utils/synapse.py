# utilities for synapse notebooks
import numpy as np


def th_u_var(lam, tau):
    """Theoretical variance for a synapse receiving uniform spikes"""
    return lam**2*(
        1./(2.*lam*tau)*(1+np.exp(-1/(lam*tau)))/(1-np.exp(-1/(lam*tau)))-1)


def appx_u_var(tau):
    """Approximate variance for synapse receiving uniform spikes

    Approximation valid for high input rates
    """
    return 1./(12.*tau**2)


def th_u_xmax(lam, tau):
    """Theoretical maximum value of synapse receiving uniform spikes"""
    return 1./(tau*(1-np.exp(-1/(lam*tau))))


def th_u_xmin(lam, tau):
    """Theoretical minimum value of synapse receiving uniform spikes"""
    return (np.exp(-1/(lam*tau)))/(tau*(1-np.exp(-1/(lam*tau))))


def th_p_var(lam, tau):
    """Theoretical variance for a synapse receiving Poisson spikes"""
    return lam/(2.*tau)
