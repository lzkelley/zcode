"""
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from numpy.testing import assert_allclose, run_module_suite

from nose.tools import assert_true, assert_false

from zcode.math import math_core


class TestMathCore(object):

    @classmethod
    def setup_class(cls):
    	np.random.seed(9865)
        cls.SIZE = 100
        cls.r1 = np.random.random(cls.SIZE)
        cls.r2 = np.random.uniform(-1.0, 1.0, size=cls.SIZE)
        # cls.y = np.random.random(100)
        # cls.v = np.random.random(100)
        # cls.X = np.random.random((100, 3))
        # cls.w = np.random.random(100)

    def test_mono(self):
        arr_g = [-1.0, 1.0, 2.0, 3.0]
        arr_ge = [-1.0, 1.0, 1.0, 2.0, 2.5]

        assert_true(math_core.mono(arr_g, 'g'))
        assert_true(math_core.mono(arr_ge, 'ge'))
        assert_true(math_core.mono(arr_g, 'ge'))
        assert_false(math_core.mono(arr_ge, 'g'))

    def test_smooth(self):
        r2 = self.r2
        ARR_SIZE = r2.size
        AMP = 10.0
        NOISE = 1.4
        SMOOTH_LENGTHS = [1, 4, 16]

        xx = np.linspace(-np.pi/4.0, 3.0*np.pi, num=ARR_SIZE)
        arrs = [AMP*np.sin(xx) + NOISE*r2
                for ii in range(len(SMOOTH_LENGTHS))]
        smArrs = [math_core.smooth(arr, smlen)
                  for (arr, smlen) in zip(arrs, SMOOTH_LENGTHS)]

        # Variance between smoothed and raw should be decreasing
        stdDiffs = [np.std(sm-arr) for sm in smArrs]
        assert stdDiffs[0] > stdDiffs[1] > stdDiffs[2]

        # Smoothing length 1 should have no effect
        assert np.all(smArrs[0] == arrs[0])

        SMOOTH_LENGTHS = [4, 16, 32]
        WIDTH = [[0.25, 0.75], 100, 100]
        LOCS  = [None, 800, 0.8]

        arr = AMP*np.sin(xx) + NOISE*r2
        smArrs = [math_core.smooth(arr, smlen, width=wid, loc=loc)
                  for smlen, wid, loc in zip(SMOOTH_LENGTHS, WIDTH, LOCS)]

        assert np.all(smArrs[0][:249] == arr[:249])
        assert np.all(smArrs[0][751:] == arr[751:])

        assert np.all(smArrs[1][:700] == arr[:700])
        assert np.all(smArrs[1][901:] == arr[901:])

        assert np.all(smArrs[2][:699] == arr[:699])
        assert np.all(smArrs[2][901:] == arr[901:])

        return

# Run all methods as if with `nosetests ...`
if __name__ == "__main__":
    run_module_suite()
