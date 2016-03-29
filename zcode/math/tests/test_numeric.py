"""Test methods for `zcode/math/math_core.py`.

Can be run with:
    $ nosetests math/tests/test_math_core.py

"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from numpy.testing import run_module_suite
from nose.tools import assert_true

from zcode.math import numeric


class TestMathCore(object):

    @classmethod
    def setup_class(cls):
        np.random.seed(9865)
        cls.SIZE = 100
        cls.r2 = np.random.uniform(-1.0, 1.0, size=cls.SIZE)

    def test_smooth(self):
        r2 = self.r2
        ARR_SIZE = r2.size
        AMP = 10.0
        NOISE = 1.4
        SMOOTH_LENGTHS = [1, 4, 16]

        xx = np.linspace(-np.pi/4.0, 3.0*np.pi, num=ARR_SIZE)
        arrs = [AMP*np.sin(xx) + NOISE*r2
                for ii in range(len(SMOOTH_LENGTHS))]
        smArrs = [numeric.smooth(arr, smlen)
                  for (arr, smlen) in zip(arrs, SMOOTH_LENGTHS)]

        # average derivative should be progressively smaller
        stdDiffs = [np.mean(np.diff(sm)) for sm in smArrs]
        assert_true(stdDiffs[0] > stdDiffs[1] > stdDiffs[2])

        # Smoothing length 1 should have no effect
        assert_true(np.all(smArrs[0] == arrs[0]))

# Run all methods as if with `nosetests ...`
if __name__ == "__main__":
    run_module_suite()
