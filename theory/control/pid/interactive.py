# This module contains functions and classes to make interactive lti plots
from abc import ABC, abstractmethod

import numpy as np
from scipy.signal import (
    impulse, impulse2,
    step, step2,
    lsim, lsim2,
    bode
)

from ipywidgets import interact
from bokeh.io import push_notebook, show
from bokeh.plotting import figure
from bokeh.layouts import gridplot, column


def analyze_lti(lti, t=np.linspace(0,10,100)):
    """Create a set of plots to analyze an lti system

    Args:
        lti (instance of an Interactive LTI system subclass ):  lti system
            to interact with
        t (numpy array): Timepoints at which to evaluate the impulse and step
            responses
    """
    t_impulse, y_impulse = impulse(lti.sys)
    fig_impulse = figure(
        title="impulse response",
        x_range=(0, 10), y_range=(-0.1, 3),
        x_axis_label='time')
    g_impulse = fig_impulse.line(x=t_impulse, y=y_impulse)

    t_step, y_step = step(lti.sys)
    fig_step = figure(
        title="step response",
        x_range=(0, 10), y_range=(-0.04, 1.04)
    )
    g_step = fig_step.line(x=t_step, y=y_step)

    fig_pz = figure(title="poles and zeroes", x_range=(-4, 1), y_range=(-2,2))
    g_poles = fig_pz.circle(
        x=lti.sys.poles.real, y=lti.sys.poles.imag, size=10, fill_alpha=0
    )
    g_rootlocus = fig_pz.line(x=[0, -10], y=[0, 0], line_color='red')

    w, mag, phase, = bode(lti.sys)
    fig_bode_mag = figure(x_axis_type="log")
    g_bode_mag = fig_bode_mag.line(w, mag)
    fig_bode_phase = figure(x_axis_type="log")
    g_bode_phase = fig_bode_phase.line(w, phase)

    fig_bode = gridplot(
        [fig_bode_mag], [fig_bode_phase], nrows=2,
        plot_width=300, plot_height=150,
        sizing_mode="fixed")

    grid = gridplot(
        [fig_impulse, fig_step],
        [fig_pz, fig_bode],
        plot_width=300, plot_height=300,
        sizing_mode="fixed"
    )

    show(grid, notebook_handle=True)

    def update(**kwargs):
        lti.update(**kwargs)
        g_impulse.data_source.data['x'], g_impulse.data_source.data['y'] = (
            impulse(lti.sys, T=t))

        g_step.data_source.data['x'], g_step.data_source.data['y'] = (
            step(lti.sys, T=t))

        g_poles.data_source.data['x'] = lti.sys.poles.real
        g_poles.data_source.data['y'] = lti.sys.poles.imag

        w, mag, phase, = bode(lti.sys)
        g_bode_mag.data_source.data['x'] = w
        g_bode_mag.data_source.data['y'] = mag
        g_bode_phase.data_source.data['x'] = w
        g_bode_phase.data_source.data['y'] = phase

        push_notebook()

    interact(update, **lti.update_kwargs)

class InteractiveLTI(ABC):
    """An abstract base class for interactive lti systems

    Provides an update method to use by the analyze_lti function
    """
    def __init__(self, sys):
        self.sys = sys
        self.update_kwargs = {}

    @abstractmethod
    def update(self, kp):
        """update function called by analyze_lti"""

class PControllerSPlant(InteractiveLTI):
    """Provides a p controller and 1/s plant

    Overall transfer function given by
           kp 
    H(s)= ----
          s+kp
    """
    def __init__(self, sys):
        super().__init__(sys)
        self.update_kwargs = {'kp':(0.1, 3., 0.1)}

    def update(self, kp):
        self.sys.poles = [-kp]
        self.sys.gain = kp
