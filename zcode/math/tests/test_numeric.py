"""Test methods for `zcode/math/math_core.py`.

Can be run with:
    $ nosetests math/tests/test_math_core.py

"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from numpy.testing import run_module_suite
from nose.tools import assert_true

import zcode
import zcode.plot
from zcode.math import numeric


class TestMathCore(object):

    @classmethod
    def setup_class(cls):
        np.random.seed(9865)
        cls.SIZE = 1000
        cls.r2 = np.random.uniform(-1.0, 1.0, size=cls.SIZE)

    def test_smooth(self):
        r2 = self.r2
        ARR_SIZE = r2.size
        AMP = 10.0
        NOISE = 1.4
        SMOOTH_LENGTHS = [1, 4, 10]
        NUM = len(SMOOTH_LENGTHS)

        xx = np.linspace(-np.pi/4.0, 3.0*np.pi, num=ARR_SIZE)
        arrs = [AMP*np.sin(xx) + NOISE*r2
                for ii in range(len(SMOOTH_LENGTHS))]
        sm_arrs = [numeric.smooth(arr, smlen)
                   for (arr, smlen) in zip(arrs, SMOOTH_LENGTHS)]

        # from matplotlib import pyplot as plt
        # fig, ax = plt.subplots()
        # colors = zcode.plot.plot_core.color_cycle(NUM)
        # for ii, (ar, sm) in enumerate(zip(arrs, sm_arrs)):
        #     ax.plot(ar, color=colors[ii], alpha=0.5)
        #     ax.plot(sm, color=colors[ii], alpha=0.5, ls='--')
        #
        # fig.savefig('test.pdf')
        # plt.close('all')

        # average derivative should be progressively smaller
        stds = [np.mean(np.diff(sm[10:-10])) for sm in sm_arrs]
        print("stds = {}".format(stds))
        assert_true(stds[0] > stds[1] > stds[2])

        # Smoothing length 1 should have no effect
        assert_true(np.all(sm_arrs[0] == arrs[0]))
        return


# Run all methods as if with `nosetests ...`
if __name__ == "__main__":
    run_module_suite()
