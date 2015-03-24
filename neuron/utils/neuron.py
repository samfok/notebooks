# define neuron models
import numpy as np
from scipy.optimize import bisect
from multiprocessing import Pool, cpu_count
from data import scalar_to_array


###############################################################################
# theoretical and theoretical approximations of input, firing rate relations ##
###############################################################################
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
    u = scalar_to_array(u)
    f = np.zeros_like(u)
    idx = u > xt
    f[idx] = (tref-tau*np.log(1.-xt/u[idx]))**-1.
    return f


def taylor1_lif_fi(a, u, tau, tref, xt, clip_subxt=False):
    """First order Taylor series approximation of the LIF tuning curve

    Parameters
    ----------
    a: float
        input value around which to approximate
    u : array-like of floats
        input
    tau : float
        membrane time constant
    tref : float
        refractory period
    xt : float
        threshold
    clip_subxt : boolean (optional)
        Whether to clip negative values in the approximation to 0
    """
    assert a > xt, "a must be > xt"
    u = scalar_to_array(u)
    k1 = tau*xt/((tref-tau*np.log(1-xt/a))**2*a*(a-xt))
    k0 = th_lif_fi(a, tau, tref, xt) - k1*a
    f = k0 + k1*u
    if clip_subxt:
        f[f < 0.] = 0.
    return f


def th_lif_if(f, tau, tref, xt):
    """Theoretically invert the LIF tuning curve

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

    Returns the input that produced the given firing rates.
    """
    f = scalar_to_array(f)
    assert (f > 0.).all(), "LIF tuning curve only invertible for f>0."
    u = xt/(1-np.exp((tref-1./f)/tau))
    return u


def _th_lif_dfdu(u, f, tau, tref, xt, out):
    out = f**2 * tau * xt / (u * (u-xt))
    return out


def th_lif_dfdu(u, tau, tref, xt, f=None, out=None):
    """Derivative of the LIF firing rate with respect to the input

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
    f : array_like of floats (optional)
        firing rate about which to compute the derivative
    out : array-like (optional)
    """
    u = scalar_to_array(u)
    if f is None:
        f = th_lif_fi(u, tau, tref, xt)
    dfdu = out
    if dfdu is None:
        dfdu = np.zeros_like(u)
    if (u > xt).all():  # faster
        dfdu = _th_lif_dfdu(u, f, tau, tref, xt, dfdu)
    else:  # handles below threshold inputs
        idx = u > xt
        if idx.any():
            dfdu[idx] = _th_lif_dfdu(u[idx], f[idx], tau, tref, xt, dfdu[idx])
        idx = u < xt
        dfdu[idx] = 0.
        idx = u == xt
        dfdu[idx] = np.nan
    return dfdu


def th_ralif_if(f, tau_m, tref, xt, af):
    """Theoretically invert the rate-based adaptive LIF tuning curve

    Parameters
    ----------
    f : array-like of floats
        firing rates (must be > 0)
    tau_m : float
        membrane time constant
    tref : float
        refractory period
    xt : float
        threshold
    af : float (optional)
        scales the feedback synapse state into a current

    Returns the input that produced the given firing rates.
    """
    f = scalar_to_array(f)
    assert (f > 0.).all(), "raLIF tuning curve only invertible for f>0."
    u_net = th_lif_if(f, tau_m, tref, xt)
    u_f = af * f
    u_in = u_net + u_f
    return u_in


def th_ralif_dudt(u_net, u_in, f, af, tau_f):
    """Derivative of the raLIF net input with respect to time"""
    dudt = 1./tau_f * (-u_net + u_in - af*f)
    return dudt


def th_ralif_dfdt(u_net, u_in, f, tau_m, tref, xt, af, tau_f, ret_dudt=False):
    """Derivative of the raLIF firing rate with respect to time"""
    dudt = th_ralif_dudt(u_net, u_in, f, af, tau_f)
    dfdu = th_lif_dfdu(u_net, tau_m, tref, xt, f=f)
    dfdt = dfdu * dudt
    if ret_dudt:
        return dfdt, dudt
    else:
        return dfdt


def th_usyn_xmin(lam, tau):
    """Theoretical steady state xmin for synapse with uniform input"""
    ret = np.zeros(len(lam))
    idx = lam > 0
    ret[idx] = np.exp(-1./(lam[idx]*tau))/(tau*(1.-np.exp(-1./(lam[idx]*tau))))
    return ret


def th_usyn_xmax(lam, tau):
    """Theoretical steady state xmax for synapse with uniform input"""
    ret = np.zeros(len(lam))
    idx = lam > 0
    ret[idx] = 1./(tau*(1.-np.exp(-1./(lam[idx]*tau))))
    return ret


###############################################################################
# numerical methods for determining input, firing rate relations ##############
###############################################################################
def _alif_u_tspk(tspk, tau_m, tref, xt, af, tau_f):
    """Computes the input u from tspk for an adaptive LIF neuron"""
    t0 = 1./(1-np.exp(-tspk/tau_m))
    if tau_m != tau_f:
        t1 = af*np.exp(-tref/tau_f)*(np.exp(-tspk/tau_f)-np.exp(-tspk/tau_m))
        t2 = (1.-np.exp(-(tref+tspk)/tau_f))*(tau_m-tau_f)
    elif tau_m == tau_f:  # because Python doesn't know LHopital's Rule
        t1 = -af*tspk*np.exp(-(tref+tspk)/tau_m)
        t2 = tau_m**2*(1-np.exp(-(tref+tspk)/tau_m))
    u = t0*(xt-t1/t2)
    return u


def num_alif_fi(u_in, tau_m, tref, xt, af, tau_f, min_f=.001, max_f=None,
                max_iter=100, tol=1e-3, verbose=False):
    """Numerically determine the approximate adaptive LIF neuron tuning curve

    Uses the bisection method (binary search in CS parlance) to find the
    steady state firing rate

    Parameters
    ----------
    u_in : array-like of floats
        input
    tau_m : float
        membrane time constant
    tref : float
        refractory period
    xt : float
        threshold
    af : float (optional)
        scales the inhibitory feedback
    tau_f : float (optional)
        time constant of the feedback synapse
    min_f : float (optional)
        minimum firing rate to consider nonzero
    max_f : float (optional)
        maximum firing rate to seed bisection method with. Be sure that the
        maximum firing rate will indeed be within this bound otherwise the
        binary search will break
    max_iter : int (optional)
        maximium number of iterations in binary search
    tol : float (optional)
        tolerance of binary search algorithm in u_in. The algorithm terminates
        when maximum difference between estimated u_in and input u_in is within
        tol
    """
    u_in = scalar_to_array(u_in)
    f_ret = np.zeros_like(u_in)
    f_high = max_f
    if max_f is None:
        f_high = 1./tref
    f_low = min_f
    tspk_high = 1./f_low - tref
    tspk_low = 1./f_high - tref

    # check for u_in that produces firing rates below the minimum firing rate
    u_min = _alif_u_tspk(tspk_high, tau_m, tref, xt, af, tau_f)
    idx = u_in > u_min  # selects the range of u_in that produces spikes
    if not idx.any():
        return f_ret
    tspk_high = np.zeros_like(u_in[idx]) + tspk_high
    tspk_low = np.zeros_like(u_in[idx]) + tspk_low

    exit_msg = 'reached max iterations'
    for i in xrange(max_iter):
        assert (tspk_low <= tspk_low).all(), 'binary search failed'
        tspk = (tspk_high+tspk_low)/2.
        uhat = _alif_u_tspk(tspk, tau_m, tref, xt, af, tau_f)
        max_diff = np.max(np.abs(u_in[idx]-uhat))
        if max_diff < tol:
            exit_msg = 'reached tolerance in %d iterations' % (i+1)
            break
        high_idx = uhat > u_in[idx]  # where our estimate of u_in is too high
        low_idx = uhat <= u_in[idx]  # where our estimate of u_in is too low
        tspk_high[low_idx] = tspk[low_idx]
        tspk_low[high_idx] = tspk[high_idx]
    f_ret[idx] = 1./(tref+tspk)
    if verbose:
        print exit_msg
    return f_ret


def scipy_alif_fi(u_in, tau_m, tref, xt, af, tau_f, method=bisect,
                  min_f=.001, max_f=None, max_iter=100):
    """Numerically determine the approximate adaptive LIF neuron tuning curve

    Same idea as num_alif_fi but uses scipy methods instead

    Parameters
    ----------
    u_in : array-like of floats
        input
    tau_m : float
        membrane time constant
    tref : float
        refractory period
    xt : float
        threshold
    af : float (optional)
        scales the inhibitory feedback
    tau_f : float (optional)
        time constant of the feedback synapse
    method : bisect or brentq (optional)
        scipy method to use
    min_f : float (optional)
        minimum firing rate to consider nonzero
    max_f : float (optional)
        maximum firing rate to seed bisection method with. Be sure that the
        maximum firing rate will indeed be within this bound otherwise the
        binary search will break
    max_iter : int (optional)
        maximium number of iterations in binary search
    tol : float (optional)
        tolerance of binary search algorithm in u_in. The algorithm terminates
        when maximum difference between estimated u_in and input u_in is within
        tol
    """
    f_ret = np.zeros_like(u_in)
    f_high = max_f
    if max_f is None:
        f_high = 1./tref
    f_low = min_f
    tspk_high = 1./f_low - tref
    tspk_low = 1./f_high - tref

    # check for u_in that produces firing rates below the minimum firing rate
    u_min = _alif_u_tspk(tspk_high, tau_m, tref, xt, af, tau_f)
    idx = u_in > u_min  # selects the range of u_in that produces spikes
    if not idx.any():
        return f_ret

    def _root_wrapper(tspk, tau_m, tref, xt, af, tau_f, u_in):
        return u_in - _alif_u_tspk(tspk, tau_m, tref, xt, af, tau_f)

    f = np.zeros_like(u_in[idx])
    for i, u_val in enumerate(u_in[idx]):
        tspk0 = method(_root_wrapper, tspk_low, tspk_high,
                       args=(tau_m, tref, xt, af, tau_f, u_val),
                       maxiter=max_iter)
        f[i] = 1./(tref+tspk0)
    f_ret[idx] = f
    return f_ret


def num_alif_fi_mu_apx(u_in, tau_m, tref, xt, af, tau_f,
                       max_iter=100, rel_tol=1e-3,
                       spiking=True, verbose=False):
    """Numerically determine the approximate adaptive LIF neuron tuning curve

    The solution is approximate because this function assumes that the
    steady state feedback value is fixed at its mean (_mu_apx stands for mu
    approximation).

    Uses the bisection method (binary search in CS parlance) to find the
    steady state firing rate

    Parameters
    ----------
    u_in : array-like of floats
        input
    tau_m : float
        membrane time constant
    tref : float
        refractory period
    xt : float
        threshold
    af : float (optional)
        scales the inhibitory feedback
    tau_f : float (optional)
        time constant of the feedback synapse
    max_iter : int (optional)
        maximium number of iterations in binary search
    rel_tol : float (optional)
        relative tolerance of binary search algorithm. The algorithm terminates
        when the maximum relative difference between the estimated u_in and the
        input u_in is within rel_tol
    spiking : bool (optional)
        If True, af is scaled to account for the refractory period.
        If False, af is used as given, which is equivalent to a rate-based
        adaptive lif neuron's behavior
    verbose : bool (optional)
        If True, prints whether algorithm finishes successfully by satisfying
        the tolerance or failed by reaching the maximum number of iterations
    """
    assert af > 0, "inhibitory feedback scaling must be > 0"
    if spiking:
        af *= np.exp(-tref/tau_f)
    u_in = scalar_to_array(u_in)

    f_high = th_lif_fi(u_in, tau_m, tref, xt)
    f_ret = np.zeros_like(u_in)
    idx = f_high > 0.
    f_high = f_high[idx]
    f_low = np.zeros_like(f_high)
    exit_msg = 'reached max iterations'
    for i in xrange(max_iter):
        f = (f_high+f_low)/2.
        u_net = th_lif_if(f, tau_m, tref, xt)
        uf = f*af
        uhat = u_net + uf
        high_idx = uhat > u_in[idx]
        low_idx = uhat <= u_in[idx]
        f_high[high_idx] = f[high_idx]
        f_low[low_idx] = f[low_idx]

        max_rel_diff = np.max(np.abs(uhat-u_in[idx])/u_in[idx])
        if max_rel_diff < rel_tol:
            exit_msg = 'reached tolerance'
            break
    f_ret[idx] = f
    if verbose:
        print exit_msg
    return f_ret


def num_ralif_fi(*args, **kwargs):
    """Numerically determine the rate-based adaptive LIF neuron tuning curve

    Same as num_alif_fi_mu_apx with spiking set to False
    See num_alif_fi_mu_apx for parameter descriptions

    Note that tau_f is not relevant to finding the rate-based adaptive LIF
    neuron's steady-state firing rate
    """
    assert len(args) < 6, (
        'num_ralif_fi only takes up to 5 positional inputs. Are you passing ' +
        'in a tau_f by accident? num_ralif_fi does not use tau_f')
    assert 'tau_f' not in kwargs, 'num_ralif_fi does not use tau_f'
    return num_alif_fi_mu_apx(*args, tau_f=None, spiking=False, **kwargs)


###############################################################################
# empirical methods for determining input, firing rate relations ##############
###############################################################################
def sim_lif_fi(dt, u, tau, tref, xt):
    """Find the LIF tuning curve by simulating the neuron

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
    # theory used to set how long to simulate
    th_f = th_lif_fi(u, tau, tref, xt)
    sim_f = np.zeros_like(th_f)
    for idx, u_val in enumerate(u):
        if th_f[idx] < .01:
            # estimated firing rate too low. would require too long to simulate
            continue
        T_f = 1./th_f[idx]  # expected interspike interval
        # run long enough to collect some spikes
        run_time = 5.*T_f
        u_in = u_val+np.zeros(int(np.ceil(run_time/dt)))
        spike_times = run_lifsoma(dt, u_in, tau, tref, xt)
        isi = np.diff(spike_times[-3:])
        if (isi[-2]-isi[-1])/isi[-2] > .01:
            print('Warning (sim_lif_fi): ' +
                  'Greater than 1% change in isi between last two isi. ' +
                  'Something is wrong for u=%.2f...' % u_val)
        sim_f[idx] = 1./isi[-1]
    return sim_f


def _sim_alif_fi_worker(args):
    return _sim_alif_fi_worker_unwrapped(*args)


def _sim_alif_fi_worker_unwrapped(dt, u_in, tau_m, tref, xt, af, tau_f):
    num_af = num_alif_fi(u_in, tau_m, tref, xt, af, tau_f)
    if num_af < .01:
        # estimated firing rate too low. would require too long to simulate
        return 0.
    T_af = 1./num_af  # expected interspike interval
    # run long enough to reach steady state and collect some spikes
    run_time = 5.*tau_f+5.*T_af
    u_in = u_in+np.zeros(int(np.ceil(run_time/dt)))
    spike_times = run_alifsoma(dt, u_in, tau_m, tref, xt, af, tau_f)
    isi = np.diff(spike_times[-3:])
    assert ((isi[-2]-isi[-1])/isi[-2] < .01), (
        'sim_alif_fi: Greater than 1% change in isi between last two isi. ' +
        'Has not reached steady state for u_in=%.2f' % u_in)
    return 1./isi[-1]


def sim_alif_fi(dt, u_in, tau_m, tref, xt, af=1e-3, tau_f=1e-2,
                max_proc=cpu_count()-1):
    """Find the adaptive LIF tuning curve by simulating the neuron

    Parameters
    ----------
    dt : float or array-like of floats
        time step. If an array, indicates dt to use with each element of u_in.
    u_in : array-like of floats
        input
    tau_m : float
        membrane time constant
    tref : float
        refractory period
    xt : float
        threshold
    af : float (optional)
        scales the inhibitory feedback
    tau_f : float (optional)
        time constant of the feedback synapse
    max_proc : int (optional)
        max number of cores to use
    """
    if isinstance(dt, (np.ndarray, list)):
        assert len(dt) == len(u_in), (
            'lengths of dt and u_in must match when dt is an array')
        args = [(dt_val, u_val, tau_m, tref, xt, af, tau_f)
                for u_val, dt_val in zip(u_in, dt)]
    else:
        args = [(dt, u_val, tau_m, tref, xt, af, tau_f) for u_val in u_in]

    if (max_proc in (0, None)) or (len(u_in) == 1):
        sim_af = map(_sim_alif_fi_worker, args)
    else:
        workers = Pool(max_proc)
        sim_af = workers.map(_sim_alif_fi_worker, args)
        workers.close()
        workers.join()
    return np.array(sim_af)


###############################################################################
# methods for running neuron models ###########################################
###############################################################################
def run_lifsoma(dt, u, tau, tref, xt, ret_state=False, flatten1=True):
    """Simulates an LIF soma(s) given an input current

    Returns the spike times of the LIF soma

    Parameters
    ----------
    dt : float
        time step (s)
    u : array-like (m x n)
        inputs for each time step
    tau : float
        time constant (s)
    xt : float
        threshold
    ret_state : boolean (optional)
        whether to also return the soma state
    flatten1 : boolean (optional)
        whether to flatten the outputs if there is only 1 neuron
    """
    nneurons = 1
    if len(u.shape) > 1:
        nneurons = u.shape[1]
    nsteps = u.shape[0]
    if nneurons == 1:
        u.shape = u.shape[0], 1

    decay = np.exp(-dt/tau)
    increment = (1-decay)

    spiketimes = [[] for i in xrange(nneurons)]
    state = np.zeros_like(u)
    refractory_time = np.zeros(nneurons)

    for i in xrange(1, nsteps):
        # update soma state with prev state and input
        state[i, :] = decay*state[i-1, :] + increment*u[i, :]
        dV = state[i, :]-state[i-1, :]

        # update refractory period assuming no spikes for now
        refractory_time -= dt

        # set voltages of neurons still in their refractory period to 0
        # and reduce voltage of neurons partway out of their ref. period
        state[i, :] *= (1-refractory_time/dt).clip(0, 1)

        # determine which neurons spike
        spiked = state[i, :] > xt
        spiked_idx = np.nonzero(spiked)[0]

        # linearly approximate time since neuron crossed spike threshold
        overshoot = (state[i, spiked] - xt) / dV[spiked]
        interp_spiketime = dt * (1-overshoot)

        for idx, spk_t in zip(spiked_idx, interp_spiketime):
            spiketimes[idx].append(spk_t+i*dt)

        # set spiking neurons' voltages to zero, and ref. time to tref
        state[i, spiked] = 0
        refractory_time[spiked] = tref + interp_spiketime

    if nneurons == 1 and flatten1:
        spiketimes = np.array(spiketimes[0])
    else:
        for idx in xrange(nneurons):
            spiketimes[idx] = np.array(spiketimes[idx])

    if ret_state:
        retval = spiketimes, state
    else:
        retval = spiketimes
    return retval


def run_alifsoma(dt, u_in, tau_m, tref, xt, af=1e-2, tau_f=1e-2,
                 ret_state=False, ret_fstate=False, flatten1=True):
    """Simulates an adaptive LIF soma(s) given an input current

    Returns the spike times of the LIF soma. Can also return the soma and
    feedback states depending on optional parameters.

    Parameters
    ----------
    dt : float
        time step (s)
    u_in : array-like (m x n)
        inputs for each time step
    tau_m : float
        soma time constant (s)
    xt : float
        threshold
    af : float (optional)
        scales the feedback synapse state into a current
    tau_f : float (optional)
        time constant of the feedback synapse
    ret_state : boolean (optional)
        whether to also return the soma state
    ret_fstate : boolean (optional)
        whether to also return the feedback synapse state
    flatten1 : boolean (optional)
        whether to flatten the outputs if there is only 1 neuron
    """
    nneurons = 1
    if len(u_in.shape) > 1:
        nneurons = u_in.shape[1]
    nsteps = u_in.shape[0]
    if nneurons == 1:
        u_in.shape = u_in.shape[0], 1

    decay = np.expm1(-dt/tau_m)+1  # expm1 higher precision version of exp
    increment = (1-decay)

    fdecay = np.expm1(-dt/tau_f)+1  # expm1 higher precision version of exp
    fincrement = (1-fdecay)

    spiketimes = [[] for i in xrange(nneurons)]
    state = np.zeros_like(u_in)
    fstate = np.zeros_like(u_in)
    refractory_time = np.zeros(nneurons)

    for i in xrange(1, nsteps):
        # update feedback with prev state
        fstate[i, :] = fdecay*fstate[i-1, :]

        # update soma state with prev state, input, and feedback
        state[i, :] = (decay*state[i-1, :] +
                       increment*(u_in[i, :] - af*fstate[i, :]))
        dV = state[i, :]-state[i-1, :]

        # update refractory period assuming no spikes for now
        refractory_time -= dt

        # set voltages of neurons still in their refractory period to 0
        # and reduce voltage of neurons partway out of their ref. period
        state[i, :] *= (1 - refractory_time / dt).clip(0, 1)

        # determine which neurons spike
        spiked = state[i, :] > xt

        # linearly approximate time since neuron crossed spike threshold
        overshoot = (state[i, spiked] - xt) / dV[spiked]
        interp_spiketime = dt * (1 - overshoot)

        # set spiking neurons' voltages to zero, and ref. time to tref
        state[i, spiked] = 0
        refractory_time[spiked] = tref + interp_spiketime

        # update feedback with current spikes
        fstate[i, :] += fincrement*spiked/dt

        # note the specific spike times
        spiked_idx = np.nonzero(spiked)[0]
        for idx, spk_t in zip(spiked_idx, interp_spiketime):
            spiketimes[idx].append(spk_t + i * dt)

    if nneurons == 1 and flatten1:
        spiketimes = np.array(spiketimes[0])
    else:
        for idx in xrange(nneurons):
            spiketimes[idx] = np.array(spiketimes[idx])

    retval = spiketimes
    if ret_state or ret_fstate:
        retval = [retval]
    if ret_state:
        retval.append(state)
    if ret_fstate:
        retval.append(fstate)
    return retval


def run_ralifsoma(dt, u_in, tau_m, tref, xt, af=1e-2, tau_f=1e-2,
                  f0=0., u0=None, ret_u=False, flatten1=True):
    """Simulates a rate-based adaptive LIF soma(s) given an input current

    Returns the rates of the rate-based adaptive LIF soma. Optionally the net
    input current.

    Parameters
    ----------
    dt : float
        time step (s)
    u_in : array-like (m x n)
        inputs for each time step
    tau_m : float
        soma time constant (s)
    xt : float
        threshold
    af : float (optional)
        scales the feedback synapse state into a current
    tau_f : float (optional)
        time constant of the feedback synapse
    f0 : array-like (n,) (optional)
        initial firing rate; also defines the initial feedback
    u0 : array-like (n,) (optional)
        initial net input
    ret_u : boolean
        if True, returns the net input current
    flatten1 : boolean
    """
    nneurons = 1
    if len(u_in.shape) > 1:
        nneurons = u_in.shape[1]
    nsteps = u_in.shape[0]
    if nneurons == 1:
        u_in.shape = u_in.shape[0], 1

    f = np.zeros_like(u_in)
    u = np.zeros_like(u_in)
    f[0, :] = f0
    if u0 is None:
        idx = f[0, :] > 0
        u[0, idx] = th_lif_if(f[0, idx], tau_m, tref, xt)
    else:
        u[0, :] = u0
    for i in xrange(1, nsteps):
        dfdt, dudt = th_ralif_dfdt(u[i-1, :], u_in[i-1, :], f[i-1, :],
                                   tau_m, tref, xt, af, tau_f, ret_dudt=True)
        u[i, :] = u[i-1, :] + dudt * dt
        f[i, :] = f[i-1, :] + dfdt * dt

    if nneurons == 1 and flatten1:
        f = f.reshape(-1)
        u = u.reshape(-1)

    if ret_u:
        return f, u
    else:
        return f
