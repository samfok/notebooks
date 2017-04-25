# utils for contraction notebook
import numpy as np
from matplotlib import pyplot as plt


def sim_linear_sys(A, dt, x0, N):
    """Simulates a linear system"""
    lin_fun = lambda x: A.dot(x)
    x = sim_sys(lin_fun, dt, x0, N)
    return x


def sim_sys(f, dt, x0, N):
    """Simulates a system using the simple Euler step

    Parameters
    ----------
    f : callable
        function that computes the dxdt as a function of x
    """
    n = x0.shape[0]
    x = np.zeros((N, n))
    x[0] = x0
    for i in xrange(N-1):
        x[i+1] = f(x[i]) * dt + x[i]
    return x
    
def phase(A, x1lim=(-1., 1), x2lim=(-1., 1.), n1=20, n2=20, ax=None):
    x1min, x1max = x1lim
    x2min, x2max = x2lim
    x1 = np.linspace(x1min, x1max, n1)
    x2 = np.linspace(x2min, x2max, n2)
    x1, x2 = np.meshgrid(x1, x2, indexing='ij')
    X = np.array([x1, x2])
    Y = np.zeros((2, n1, n2))
    for i in xrange(n1):
        for j in xrange(n2):
            Y[:, i, j] = A.dot(X[:, i, j])
    y1, y2 = Y

    ret_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
        ret_fig = True
    ax.quiver(x1, x2, y1, y2, units='xy', angles='xy')
    ax.set_xlim(x1lim)
    ax.set_ylim(x2lim)
    ax.axhline(0, c='k', alpha=.2)
    ax.axvline(0, c='k', alpha=.2)
    if ret_fig:
        return fig, ax
    else:
        return ax


def plot_linear_contraction(delta_x, dfdx, dt):
    """Assumes df/dx is constant"""
    N = delta_x.shape[0]
    t = np.arange(N)*dt

    evals = np.sort(np.linalg.eigvals(dfdx))

    dist_sq = np.sum(delta_x*delta_x, axis=1)
    ddist_sq_th = np.zeros(N)
    ddist_sq_ub = np.zeros(N)
    ddist_sq_lb = np.zeros(N)
    for i in xrange(N):
        dx = delta_x[i].reshape((2, 1))
        ddist_sq_th[i] = 2.*dx.T.dot(dfdx).dot(dx)
        ddist_sq_ub[i] = 2.*evals[1]*dx.T.dot(dx)
        ddist_sq_lb[i] = 2.*evals[0]*dx.T.dot(dx)
    ddist_sq_num = np.diff(dist_sq)/dt
    dx0 = delta_x[0]
    dx0_norm = dx0.dot(dx0)
    dist_sq_ub = dx0_norm * np.exp(2.*evals[1]*t)
    dist_sq_lb = dx0_norm * np.exp(2.*evals[0]*t)

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 3))
    ax1.plot(t, dist_sq, 'r', label='observed')
    ax1.plot(t, dist_sq_lb, 'b', label='theory lower bound')
    ax1.plot(t, dist_sq_ub, 'k', label='theory upper bound')
    ax1.legend(loc='best')
    ax1.set_xlabel(r'$t$', fontsize=16)
    ax1.set_title(r'$\delta\mathbf{x}^T\delta\mathbf{x}$', fontsize=16)

    ax2.plot(t[1:], ddist_sq_num, 'r', lw=1, label='observed')
    ax2.plot(t, ddist_sq_th, 'g', lw=2, label='theory')
    ax2.plot(t, ddist_sq_lb, 'b', label='theory lower bound')
    ax2.plot(t, ddist_sq_ub, 'k', label='theory upper bound')
    ax2.legend(loc='best')
    ax2.set_xlabel(r'$t$', fontsize=16)
    ax2.set_title(r'$\frac{d}{dt}(\delta\mathbf{x}^T\delta\mathbf{x})$',
                  fontsize=16, y=1.03)


def plot_nonlinear_contraction(delta_x, x, dfdx, dt):
    """Assumes df/dx is callable"""
    N = delta_x.shape[0]
    t = np.arange(N)*dt

    dist_sq = np.sum(delta_x*delta_x, axis=1)
    ddist_sq_th = np.zeros(N)
    ddist_sq_ub = np.zeros(N)
    ddist_sq_lb = np.zeros(N)
    evals = np.zeros((N, 2))
    for i in xrange(N):
        dx = delta_x[i].reshape((2, 1))
        evals[i] = np.sort(np.linalg.eigvals(dfdx(x[i])))
        ddist_sq_th[i] = 2.*dx.T.dot(dfdx(x[i])).dot(dx)
        ddist_sq_ub[i] = 2.*evals[i, 1]*dx.T.dot(dx)
        ddist_sq_lb[i] = 2.*evals[i, 0]*dx.T.dot(dx)
    ddist_sq_num = np.diff(dist_sq)/dt

    dx0 = delta_x[0]
    dx0_norm = dx0.dot(dx0)
    dist_sq_ub = np.zeros(N)
    dist_sq_lb = np.zeros(N)
    dist_sq_ub[0] = dx0_norm
    dist_sq_lb[0] = dx0_norm
    from scipy.integrate import cumtrapz
    e1_ref = cumtrapz(2*evals[:, 1], t, initial=0.)
    e0_ref = cumtrapz(2*evals[:, 0], t, initial=0.)
    for i in xrange(1, N):
        dist_sq_ub[i] = dx0_norm*np.exp(e1_ref[i])
        dist_sq_lb[i] = dx0_norm*np.exp(e0_ref[i])

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 3))
    ax1.plot(t, dist_sq, 'r', label='observed')
    ax1.plot(t, dist_sq_lb, 'b', label='theory lower bound')
    ax1.plot(t, dist_sq_ub, 'k', label='theory upper bound')
    ax1.legend(loc='best')
    ax1.set_xlabel(r'$t$', fontsize=16)
    ax1.set_title(r'$\delta\mathbf{x}^T\delta\mathbf{x}$', fontsize=16)

    ax2.plot(t[1:], ddist_sq_num, 'r', lw=1, label='observed')
    ax2.plot(t, ddist_sq_th, 'g', lw=2, label='theory')
    ax2.plot(t, ddist_sq_lb, 'b', label='theory lower bound')
    ax2.plot(t, ddist_sq_ub, 'k', label='theory upper bound')
    ax2.legend(loc='best')
    ax2.set_xlabel(r'$t$', fontsize=16)
    ax2.set_title(r'$\frac{d}{dt}(\delta\mathbf{x}^T\delta\mathbf{x})$',
                  fontsize=16, y=1.03)
