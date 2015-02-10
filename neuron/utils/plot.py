# plotting utility functions
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as mcm
from numpy import ones


def make_red_cmap(low=0., high=1., name='red'):
    """Creates a simple red colormap"""
    cdict = {'red': [(0.0, low, low),
                     (1.0, high, high)],
             'green': [(0.0, 0.0, 0.0),
                       (1.0, 0.0, 0.0)],
             'blue': [(0.0, 0.0, 0.0),
                      (1.0, 0.0, 0.0)]}
    cmap = LinearSegmentedColormap(name, cdict)
    return cmap


def make_blue_cmap(low=0., high=1., name='blue'):
    """Creates a simple blue colormap"""
    cdict = {'red': [(0.0, 0.0, 0.0),
                     (1.0, 0.0, 0.0)],
             'green': [(0.0, 0.0, 0.0),
                       (1.0, 0.0, 0.0)],
             'blue': [(0.0, low, low),
                      (1.0, high, high)]}
    cmap = LinearSegmentedColormap(name, cdict)
    return cmap


def make_color_cycle(values, cmap):
    """Generates a list of colors from a colormap cmap for values"""
    cNorm = colors.Normalize(vmin=min(values), vmax=max(values))
    scalarMap = mcm.ScalarMappable(norm=cNorm, cmap=cmap)
    color_cycle = [scalarMap.to_rgba(value) for value in values]
    return color_cycle


def _get_fig_ax(fig=None, ax=None, figp={}, axp={}, subplotp=(1, 1, 1)):
    """Handles getting or creating a figure and axes handle"""
    if fig is None:
        if ax is None:
            fig = figure(**figp)
        else:
            fig = ax.figure
    if ax is None:
        ax = fig.add_subplot(*subplotp, **axp)
    return fig, ax


def _set_ax(ax, axhline=None, axhlinep={}, axvline=None, axvlinep={},
            xlim=None, ylim=None,
            xticks=None, yticks=None, xticklabels=None, yticklabels=None,
            xlabel=None, xlabelp={}, ylabel=None, ylabelp={},
            title=None, titlep={},
            legendp={}):
    """Sets a bunch of axes properties with a single function call"""
    if axhline is not None:
        ax.axhline(axhline, **axhlinep)
    if axvline is not None:
        ax.axvline(axvline, **axvlinep)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels)
    if yticklabels is not None:
        ax.set_yticklabels(yticklabels)
    if len(legendp) != 0:
        ax.legend(**legendp)
    if xlabel is not None:
        ax.set_xlabel(xlabel, **xlabelp)
    if ylabel is not None:
        ax.set_ylabel(ylabel, **ylabelp)
    if title is not None:
        ax.set_title(title, **titlep)
    return ax


def save_close_fig(fig, fname, ax, close=False, savep={}):
    """Handles the saving and closing of a figure in a single function call"""
    if fname is not None:
        fig.savefig(fname, **savep)
    if close:
        plt.close(fig)
    else:
        return fig, ax


def plot_spike_raster(spike_times, fig=None, figp={},
                      ax=None, axp={'frameon': False}, subplotp=(1, 1, 1),
                      fname=None, close=False, savep={}, **axkwargs):
    fig, ax = _get_fig_ax(fig, ax, figp, axp, subplotp)
    ax.stem(spike_times, ones(spike_times.shape))
    _set_ax(ax, **axkwargs)
    return save_close_fig(fig, fname, ax, close, savep)


def plot_continuous(u, a, plotp={}, fig=None, figp={},
                    ax=None, axp={}, subplotp=(1, 1, 1),
                    fname=None, close=False, savep={}, **axkwargs):
    fig, ax = _get_fig_ax(fig, ax, figp, axp, subplotp)
    ax.plot(u, a, **plotp)
    _set_ax(ax, **axkwargs)
    return save_close_fig(fig, fname, ax, close, savep)


def plot_histogram(x, histp={}, fig=None, figp={},
                   ax=None, axp={}, subplotp=(1, 1, 1),
                   fname=None, close=False, savep={}, **axkwargs):
    fig, ax = _get_fig_ax(fig, ax, figp, axp, subplotp)
    ax.hist(x, **histp)
    _set_ax(ax, **axkwargs)
    return save_close_fig(fig, fname, ax, close, savep)


def plot_contour(x, y, z, contourfp={}, contourp={}, fig=None, figp={},
                 ax=None, axp={}, subplotp=(1, 1, 1),
                 fname=None, close=False, savep={}, **axkwargs):
    fig, ax = _get_fig_ax(fig, ax, figp, axp, subplotp)
    cs = ax.contourf(x, y, z, **contourfp)
    cs2 = ax.contour(cs, **contourp)
    cbar = fig.colorbar(cs)
    cbar.add_lines(cs2)
    _set_ax(ax, **axkwargs)
    return save_close_fig(fig, fname, ax, close, savep)


def plot_scatter(x, y, scatterp={}, fig=None, figp={},
                 ax=None, axp={}, subplotp=(1, 1, 1),
                 fname=None, close=False, savep={}, **axkwargs):
    fig, ax = _get_fig_ax(fig, ax, figp, axp, subplotp)
    ax.scatter(x, y, **scatterp)
    _set_ax(ax, **axkwargs)
    return save_close_fig(fig, fname, ax, close, savep)


def match_xlims(axes, lims=None):
    """matches the xlims of the axes in a list of axis"""
    if lims is not None:
        for ax in axes:
            ax.set_xlim(lims)
        return
    xlim_min, xlim_max = axes[0].get_xlim()
    for ax in axes[1:]:
        l, u = ax.get_xlim()
        xlim_min = min(l, xlim_min)
        xlim_max = max(u, xlim_max)
    for ax in axes:
        ax.set_xlim(xlim_min, xlim_max)
