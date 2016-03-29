"""Test methods for `zcode/math/math_core.py`.

Can be run with:
    $ nosetests math/tests/test_math_core.py

"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from numpy.testing import run_module_suite
import scipy as sp
import scipy.stats
from nose.tools import assert_true, assert_false, assert_equal, assert_raises

from zcode.math import math_core


class TestMathCore(object):

    @classmethod
    def setup_class(cls):
        np.random.seed(9865)
        cls.SIZE = 100
        cls.r1 = np.random.random(cls.SIZE)
        cls.r2 = np.random.uniform(-1.0, 1.0, size=cls.SIZE)

    def test_spacing(self):
        from zcode.math.math_core import spacing

        # Linear Spacing
        ref1 = np.linspace(0.0, 1.0, num=20)
        spc1 = spacing([0.0, 1.0], scale='lin', num=20)
        assert_true(np.allclose(ref1, spc1))

        # Logarithmic Spacing
        ref2 = np.logspace(0.0, 2.5, num=20)
        spc2 = spacing([np.power(10.0, 0.0), np.power(10.0, 2.5)], scale='log', num=20)
        assert_true(np.allclose(ref2, spc2))

        # Automatically selects appropriate Range
        ref3 = np.logspace(1.0, 2.0, num=13)
        spc3 = spacing([-10.0, 100.0, 0.0, 10.0], scale='log', num=13)
        assert_true(np.allclose(ref3, spc3))

        # Manually selects appropraite range
        ref4 = np.linspace(-5.0, -2.5, num=27)
        spc4 = spacing([3.0, -2.5, -5.0, 0.0], scale='lin', num=27, filter='<')
        assert_true(np.allclose(ref4, spc4))
        return

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

    def test_really1d(self):
        from zcode.math import really1d
        assert_true(really1d([1, 2, 3]))
        assert_true(really1d([1]))
        assert_true(really1d([]))
        assert_true(really1d(np.arange(10)))

        assert_false(really1d(1))
        assert_false(really1d([[1]]))
        assert_false(really1d([[1, 2], [2, 3]]))
        assert_false(really1d([[1, 2, 3], [4, 5]]))
        assert_false(really1d(np.random.random((4, 3))))
        assert_false(really1d([[]]))

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
        assert_raises(ValueError, math_core.argextrema, np.arange(4).reshape(2, 2), 'max')
        assert_raises(ValueError, math_core.argextrema, 0.0, 'max')
        #    Invalid `type` argument
        assert_raises(ValueError, math_core.argextrema, [1, 2], 'mex')
        #    Invalid `filter` argument
        assert_raises(ValueError, math_core.argextrema, [1, 2], 'max', 'e')
        #    Invalid `filter` argument
        assert_raises(ValueError, math_core.argextrema, [1, 2], 'max', 'greater')

    def test_asBinEdges_1d(self):
        print("TestMathCore.test_asBinEdges_1d")
        from zcode.math import asBinEdges, spacing

        data_1d = np.random.random(40)
        bins_1d = np.arange(20)
        # Preserves valid bins
        assert_true(np.allclose(bins_1d, asBinEdges(bins_1d, data_1d)))

        # Constructs valid bins
        #    lin
        lin_1d = spacing(data_1d, scale='lin', num=8+1)
        lin_edges_1d = asBinEdges(8, data_1d, scale='lin')
        assert_true(np.allclose(lin_1d, lin_edges_1d))
        #    log
        log_1d = spacing(data_1d, scale='log', num=7+1)
        log_edges_1d = asBinEdges(7, data_1d, scale='log')
        assert_true(np.allclose(log_1d, log_edges_1d))

        # Raises appropriate errors
        data_2d = data_1d.reshape(8, 5)
        bins_2d = bins_1d.reshape(4, 5)
        #    1D bins, 2D data
        assert_raises(ValueError, asBinEdges, bins_1d, data_2d)
        #    2D bins, 1D data
        assert_raises(ValueError, asBinEdges, bins_2d, data_1d)

    def test_asBinEdges_nd(self):
        print("TestMathCore.test_asBinEdges_nd")
        from zcode.math import asBinEdges

        data_2d = np.random.random((8, 2))
        bins_2d = np.arange(8).reshape(2, 4)
        bins_2d2 = [[0.0, 1.0], [0.0, 0.5, 1.0]]

        # Preserves valid bins
        edges_2d = asBinEdges(bins_2d, data_2d)
        assert_true(np.allclose(bins_2d, edges_2d))
        edges_2d2 = asBinEdges(bins_2d2, data_2d)
        assert_true(np.allclose(bins_2d2[0], edges_2d2[0]))
        assert_true(np.allclose(bins_2d2[1], edges_2d2[1]))

        # Constructs valid bins
        #    lin
        lin_2d1 = sp.stats.binned_statistic_dd(data_2d, None, 'count', bins=4).bin_edges
        lin_edges_2d1 = asBinEdges(4, data_2d, scale='lin')
        assert_true(np.allclose(lin_2d1, lin_edges_2d1))
        lin_2d2 = sp.stats.binned_statistic_dd(data_2d, None, 'count', bins=[4, 3]).bin_edges
        lin_edges_2d2 = asBinEdges([4, 3], data_2d, scale='lin')
        assert_true(np.allclose(lin_2d2[0], lin_edges_2d2[0]))
        assert_true(np.allclose(lin_2d2[1], lin_edges_2d2[1]))

        # Raises appropriate errors
        #    1D bins, 2D data
        assert_raises(ValueError, asBinEdges, [4], data_2d)
        #    2D bins, 1D data
        assert_raises(ValueError, asBinEdges, [4, 3, 2], data_2d)

    def test_comparison_function(self):
        from zcode.math.math_core import _comparison_function

        comp = ['g', '>']
        arr = [0.5, 1.5, -0.5, 0.0]
        res = [True, True, False, False]
        for cc in comp:
            func = _comparison_function(cc, value=0.0)
            assert_true(np.all(np.equal(func(arr), res)))

        comp = ['ge', '>=']
        arr = [0.5, 1.5, -0.5, 0.0]
        res = [True, True, False, True]
        for cc in comp:
            func = _comparison_function(cc, value=0.0)
            assert_true(np.all(np.equal(func(arr), res)))

        comp = ['l', '<']
        arr = [-10.5, -1.5, 0.5, 0.0]
        res = [True, True, False, False]
        for cc in comp:
            func = _comparison_function(cc, value=0.0)
            assert_true(np.all(np.equal(func(arr), res)))

        comp = ['le', '<=']
        arr = [-10.5, -1.5, 0.5, 0.0]
        res = [True, True, False, True]
        for cc in comp:
            func = _comparison_function(cc, value=0.0)
            assert_true(np.all(np.equal(func(arr), res)))

        comp = ['e', '=', '==']
        arr = [-10.5, 0.5, 0.0]
        res = [False, False, True]
        for cc in comp:
            func = _comparison_function(cc, value=0.0)
            assert_true(np.all(np.equal(func(arr), res)))

        comp = ['ne', '!=']
        arr = [-10.5, 0.5, 0.0]
        res = [True, True, False]
        for cc in comp:
            func = _comparison_function(cc, value=0.0)
            assert_true(np.all(np.equal(func(arr), res)))

        return

    def test_comparison_filter(self):
        from zcode.math.math_core import comparison_filter

        comp = ['g', '>']
        arr = [0.5, -1.0, 1.5, -0.5, 0.0]
        res = [0.5, 1.5]
        inds = [0, 2]
        for cc in comp:
            vals = comparison_filter(arr, cc, value=0.0)
            assert_true(np.all(np.equal(vals, res)))
            vals = comparison_filter(arr, cc, inds=True, value=0.0)
            assert_true(np.all(np.equal(vals[0], inds)))

        comp = ['le', '<=']
        arr = [0.5, -1.0, 1.5, -0.5, 0.0]
        res = [-1.0, -0.5, 0.0]
        inds = [1, 3, 4]
        for cc in comp:
            vals = comparison_filter(arr, cc, value=0.0)
            assert_true(np.all(np.equal(vals, res)))
            vals = comparison_filter(arr, cc, inds=True, value=0.0)
            assert_true(np.all(np.equal(vals[0], inds)))

        return


# Run all methods as if with `nosetests ...`
if __name__ == "__main__":
    run_module_suite()
