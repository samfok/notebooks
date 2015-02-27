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


def th_p_var(lam, tau):
    """Theoretical variance for a synapse receiving Poisson spikes"""
    return lam/(2.*tau)
