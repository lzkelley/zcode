"""
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from numpy.testing import run_module_suite
from nose.tools import assert_true, assert_false, assert_equal, assert_raises

from zcode.math import math_core


class TestMathCore(object):

    @classmethod
    def setup_class(cls):
        np.random.seed(9865)
        cls.SIZE = 100
        cls.r1 = np.random.random(cls.SIZE)
        cls.r2 = np.random.uniform(-1.0, 1.0, size=cls.SIZE)

    def test_mono(self):
        arr_g = [-1.0, 1.0, 2.0, 3.0]
        arr_ge = [-1.0, 1.0, 1.0, 2.0, 2.5]
        arr_l = [11.5, 9.2, -2.0, -301.0]
        arr_le = [11.5, 9.2, -2.0, -2.0, -301.0]
        arr_e = 11*[1.0]

        assert_true(math_core.mono(arr_g, 'g'))
        assert_true(math_core.mono(arr_ge, 'ge'))
        assert_true(math_core.mono(arr_g, 'ge'))
        assert_false(math_core.mono(arr_ge, 'g'))

        assert_true(math_core.mono(arr_l, 'l'))
        assert_true(math_core.mono(arr_le, 'le'))
        assert_true(math_core.mono(arr_l, 'le'))
        assert_false(math_core.mono(arr_le, 'l'))

        assert_true(math_core.mono(arr_e, 'e'))
        assert_false(math_core.mono(arr_le, 'e'))

    def test_argextrema(self):
        # Basic usage without filtering
        assert_equal(math_core.argextrema([-1, -5, 2, 10], 'min'), 1)
        assert_equal(math_core.argextrema([-1, -5, 2, 10], 'max'), 3)

        # Filtering
        #    min
        assert_equal(math_core.argextrema([-1, -5, 2, 10, 0], 'min', 'g'), 2)
        assert_equal(math_core.argextrema([-1, -5, 2, 10, 0], 'min', 'ge'), 4)
        assert_equal(math_core.argextrema([-1, -5, 0, 2, 10], 'min', 'l'), 1)
        assert_equal(math_core.argextrema([-1, -5, 0, 2, 10], 'min', 'le'), 1)
        #    max
        assert_equal(math_core.argextrema([-1, -5, 2, 10, 0], 'max', 'g'), 3)
        assert_equal(math_core.argextrema([-1, -5, 2, 10, 0], 'max', 'ge'), 3)
        assert_equal(math_core.argextrema([-1, -5, 0, 2, 10], 'max', 'l'), 0)
        assert_equal(math_core.argextrema([-1, -5, 0, 2, 10], 'max', 'le'), 2)

        # Raises appropriate errors
        #    Incorrect shape input array
        assert_raises(ValueError, math_core.argextrema(np.arange(4).reshape(2, 2), 'max'))
        assert_raises(ValueError, math_core.argextrema(0.0, 'max'))
        #    Invalid `type` argument
        assert_raises(ValueError, math_core.argextrema([1, 2], 'mex'))
        #    Invalid `filter` argument
        assert_raises(ValueError, math_core.argextrema([1, 2], 'max', 'e'))
        #    Invalid `filter` argument
        assert_raises(ValueError, math_core.argextrema([1, 2], 'max', 'greater'))



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

        # average derivative should be progressively smaller
        stdDiffs = [np.mean(np.diff(sm)) for sm in smArrs]
        assert stdDiffs[0] > stdDiffs[1] > stdDiffs[2]

        # Smoothing length 1 should have no effect
        assert np.all(smArrs[0] == arrs[0])

        '''
        SMOOTH_LENGTHS = [1, 4, 8]
        twid = np.int(np.floot(ARR_SIZE/4))
        WIDTH = [[0.25, 0.75], 2*twid]
        LOCS  = [None, 0.5]

        arr = AMP*np.sin(xx) + NOISE*r2
        smArrs = [math_core.smooth(arr, smlen, width=wid, loc=loc)
                  for smlen, wid, loc in zip(SMOOTH_LENGTHS, WIDTH, LOCS)]

        assert np.all(smArrs[0][:twid-2] == arr[:twid-2])
        assert np.all(smArrs[0][-twid-2:] == arr[-twid-2:])

        assert np.all(smArrs[1][:700] == arr[:700])
        assert np.all(smArrs[1][901:] == arr[901:])
        '''

        return

# Run all methods as if with `nosetests ...`
if __name__ == "__main__":
    run_module_suite()
