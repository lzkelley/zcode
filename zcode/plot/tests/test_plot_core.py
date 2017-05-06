"""Test methods for `zcode/plot/plot_core.py`.

Can be run with:
    $ nosetests plot/tests/test_plot_core.py

"""
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import numpy as np
import matplotlib.pyplot as plt
import warnings
from numpy.testing import run_module_suite
from nose.tools import assert_true

from zcode.plot import plot_core
import zcode.inout as zio

_THIS_PATH = os.path.abspath(os.path.dirname(__file__))
_PLOT_DIR = os.path.join(_THIS_PATH, 'figures/')


class TestPlotCore(object):

    @classmethod
    def setup_class(cls):
        np.random.seed(13243546)
        if not os.path.exists(_PLOT_DIR):
            os.makedirs(_PLOT_DIR)

        if not os.path.isdir(_PLOT_DIR):
            raise RuntimeError("Plot output path '{}' is not a directory!".format(_PLOT_DIR))

        pass

    def test_color_set(self):
        print("TestPlotCore.test_color_set()")
        fname = os.path.join(_PLOT_DIR, 'test_color_set.pdf')

        xkcd_colors = plot_core._COLOR_SET_XKCD
        max_num = len(xkcd_colors)
        NROWS = 4

        xx = np.linspace(-np.pi/4, 4*np.pi, 100)
        # phi = np.random.uniform(0.0, np.pi, max_num)
        phi = np.linspace(0.0, 2*np.pi, max_num)
        omega = np.random.uniform(1.0, 2.0, max_num)
        # omega = np.ones(max_num)

        fig, ax = plt.subplots(figsize=[12, 18], nrows=NROWS)
        plt.subplots_adjust(left=0.05, right=0.75, top=0.97, bottom=0.03, hspace=0.1)
        colors = plot_core.color_set(max_num)
        lines = []
        names = []
        for ii, (cc, nn) in enumerate(zip(colors, xkcd_colors)):
            label = "{:3d}: '{}'".format(ii, nn)
            for jj in range(NROWS):
                my_max = max_num*(jj+1)/NROWS
                if ii < my_max:
                    ll, = ax[jj].plot(np.sin(omega[ii] * xx + phi[ii]), ls='-', lw=5.0, color=cc,
                                      alpha=0.8, label=label)
                if ii == max_num-1:
                    ax[jj].set_title("{:3.0f} Colors".format(my_max-1))

            lines.append(ll)
            names.append(label)

        # Try to add legend using `plot_core.legend`, but catch failure and fall back to builtin
        #    since `legend` isnt the focus of this test
        try:
            plot_core.legend(fig, lines, names, x=0.99, y=0.5, halign='right', valign='center',
                             framealpha=0.5)
        except Exception as err:
            warnings.warn("`plot_core.legend` failed: '{}'".format(str(err)))
            plt.legend(framealpha=0.5)

        # Save figure and print where
        fig.savefig(fname)
        warnings.warn("Saved figure to '{}'".format(fname))
        return


# Run all methods as if with `nosetests ...`
if __name__ == "__main__":
    run_module_suite()
