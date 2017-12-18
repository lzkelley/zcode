"""Test methods for `zcode/math/math_core.py`.

Can be run with:
    $ nosetests math/tests/test_math_core.py
    $ nosetests math/tests/test_math_core.py:TestMathCore.test_around
    $ python math/tests/test_math_core.py

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

    def test_argnearest_ordered(self):
        from zcode.math.math_core import argnearest
        xx = np.array([0.2, 0.8, 1.3, 1.5, 2.0, 3.1, 3.8, 3.9, 4.5, 5.1, 5.5])
        yy = np.array([-1, 0.2, 1, 1.4, 2, 3, 4, 5, 5.5, 10])
        correct = [0, 0, 1, 2, 4, 5, 7, 9, 10, 10]
        retval = argnearest(xx, yy, assume_sorted=True)
        assert_true(np.all(correct == retval))
        print("Options = {}".format(xx))
        print("Targets = {}".format(yy))
        print("retval  = {}".format(retval))
        print("correct = {}".format(correct))
        return

    def test_argnearest_unordered_x(self):
        from zcode.math.math_core import argnearest
        xx = np.array([0.2, 0.8, 1.3, 1.5, 2.0, 3.1, 3.8, 3.9, 4.5, 5.1, 5.5])
        yy = np.array([-1, 0.2, 1, 1.4, 2, 3, 4, 5, 5.5, 10])
        correct = np.array([2, 2, 8, 7, 0, 6, 9, 3, 4, 4])

        # ix = np.random.permutation(xx.size)
        ix = np.array([4,  3,  0,  9, 10,  8,  5,  2,  1,  7,  6])
        xx = xx[ix]

        retval = argnearest(xx, yy, assume_sorted=False)
        print("Options = {}".format(xx))
        print("Targets = {}".format(yy))
        print("retval  = {}".format(retval))
        print("nearest = {}".format(xx[retval]))
        print("Targets = {}".format(yy))
        print("retval  = {}".format(retval))
        print("correct = {}".format(correct))
        assert_true(np.all(correct == retval))
        return

    def test_argnearest_unordered_xy(self):
        from zcode.math.math_core import argnearest
        xx = np.array([0.2, 0.8, 1.3, 1.5, 2.0, 3.1, 3.8, 3.9, 4.5, 5.1, 5.5])
        yy = np.array([-1, 0.2, 1, 1.4, 2, 3, 4, 5, 5.5, 10])
        correct = np.array([0, 7, 6, 3, 4, 9, 2, 4, 2, 8])

        # ix = np.random.permutation(xx.size)
        ix = np.array([4,  3,  0,  9, 10,  8,  5,  2,  1,  7,  6])
        xx = xx[ix]
        iy = np.array([4, 3, 5, 7, 9, 6, 0, 8, 1, 2])
        yy = yy[iy]

        retval = argnearest(xx, yy, assume_sorted=False)
        print("Options = {}".format(xx))
        print("Targets = {}".format(yy))
        print("retval  = {}".format(retval))
        print("nearest = {}".format(xx[retval]))
        print("Targets = {}".format(yy))
        print("retval  = {}".format(retval))
        print("correct = {}".format(correct))
        assert_true(np.all(correct == retval))
        return

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

        # Only integral (whole number) values
        # log spacing
        vals = [2.34, 365.23]
        res = np.array([2., 3., 4., 5., 6., 7., 8., 9., 10.,
                        20., 30., 40., 50., 60., 70., 80., 90., 100.,
                        200., 300., 400.])
        retvals = spacing(vals, 'log', integers=True)
        print("integers, log\n", vals, "\n\t", res, "\n\t", retvals)
        print(retvals)
        print(np.allclose(retvals, res))
        assert_true(np.allclose(retvals, res))

        # lin spacing
        vals = [2.34, 11.23]
        res = np.arange(2, 13)
        retvals = spacing(vals, 'lin', integers=True)
        print("integers, lin\n", vals, "\n\t", res, "\n\t", retvals)
        print(np.allclose(retvals, res))
        assert_true(np.allclose(retvals, res))

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

    def test_ordered_groups(self):
        arr = np.array([99, 77, 14, 21, 71, 64, 98, 38, 66, 25])
        sinds = np.argsort(arr)
        targets = [40, 77]
        print("arr = {}, targets = {}, sorted arr = {}".format(arr, targets, arr[sinds]))

        # Group into elements below targets
        #    Exclusively
        print("Below, exclusive:")
        locs, isort = math_core.ordered_groups(arr, targets, inds=None, dir='b', include=False)
        assert_true(np.all(sinds == isort))
        #    Check subsets from each target location
        for ll, tt in zip(locs, targets):
            print("target = {}, loc = {}".format(tt, ll))
            print(set(arr[isort[:ll]]), set(arr[sinds][arr[sinds] < tt]))
            assert_true(set(arr[isort[:ll]]) == set(arr[sinds][arr[sinds] < tt]))
        #    Inclusively
        print("Below, inclusive:")
        locs, isort = math_core.ordered_groups(arr, targets, inds=None, dir='b', include=True)
        assert_true(np.all(sinds == isort))
        #    Check subsets from each target location
        for ll, tt in zip(locs, targets):
            print("target = {}, loc = {}".format(tt, ll))
            print(set(arr[isort[:ll]]), set(arr[sinds][arr[sinds] <= tt]))
            assert_true(set(arr[isort[:ll]]) == set(arr[sinds][arr[sinds] <= tt]))

        # Group into elements above targets
        #    Exclusive
        print("Above, exclusive:")
        locs, isort = math_core.ordered_groups(arr, targets, inds=None, dir='a', include=False)
        assert_true(np.all(sinds[::-1] == isort))
        # Check subsets from each target location
        for ll, tt in zip(locs, targets):
            print("target = {}, loc = {}".format(tt, ll))
            print(set(arr[isort[:ll]]), set(arr[sinds][arr[sinds] > tt]))
            assert_true(set(arr[isort[:ll]]) == set(arr[sinds][arr[sinds] > tt]))

        #    Exclusive
        print("Above, inclusive:")
        locs, isort = math_core.ordered_groups(arr, targets, inds=None, dir='a', include=True)
        assert_true(np.all(sinds[::-1] == isort))
        # Check subsets from each target location
        for ll, tt in zip(locs, targets):
            print("target = {}, loc = {}".format(tt, ll))
            print(set(arr[isort[:ll]]), set(arr[sinds][arr[sinds] >= tt]))
            assert_true(set(arr[isort[:ll]]) == set(arr[sinds][arr[sinds] >= tt]))

        # Should raise error for unsorted `targets`
        assert_raises(ValueError, math_core.ordered_groups, arr, targets[::-1])

        # Should raise error for `dir` not starting with 'a' or 'b'
        assert_raises(ValueError, math_core.ordered_groups, arr, targets, None, 'c')
        return

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

    def test_around(self):
        from zcode.math.math_core import around
        vals = [
            # Nearest
            #    linear
            [[123.4678, 0, 'lin', 'near'], 123.00],
            [[123.4678, 1, 'linear', 'nearest'], 123.50],
            [[123.4678, 2, 'lin', 'n'], 123.47],
            #    logarithmic
            [[123.4678, 0, 'log', 'nearest'], 100.0],
            [[123.4678, 1, 'logarithmic', 'nearest'], 120.0],
            [[123.4678, 2, 'log', 'nearest'], 123.0],
            [[123.4678, 3, 'log', 'nearest'], 123.5],
            #       Negative decimals (order-of-magnitude rounding)
            [[213.4678, -1, 'log', 'nearest'], 100.0],
            # Ceiling (up)
            #    linear
            [[123.4678, 0, 'lin', 'c'], 124.0],
            [[123.4678, 1, 'linear', 'ceiling'], 123.5],
            [[123.4678, 2, 'lin', 'ceil'], 123.47],
            #    logarithmic
            [[123.4678, 0, 'log', 'c'], 200.0],
            [[123.4678, 1, 'logarithmic', 'c'], 130.0],
            [[123.4678, 2, 'log', 'c'], 124.0],
            [[123.4678, 3, 'log', 'c'], 123.5],
            #       Negative decimals (order-of-magnitude rounding)
            [[213.4678, -1, 'log', 'c'], 1000.0],
            # Floor (down)
            #    linear
            [[123.4678, 0, 'lin', 'f'], 123.0],
            [[123.4678, 1, 'linear', 'fl'], 123.4],
            [[123.4678, 2, 'lin', 'floor'], 123.46],
            #    logarithmic
            [[123.4678, 0, 'log', 'f'], 100.0],
            [[123.4678, 1, 'logarithmic', 'f'], 120.0],
            [[123.4678, 2, 'log', 'f'], 123.0],
            [[123.4678, 3, 'log', 'f'], 123.4],
            #       Negative decimals (order-of-magnitude rounding)
            [[213.4678, -1, 'log', 'f'], 100.0],
        ]
        for vv in vals:
            print(vv)
            res = around(*vv[0])
            print("\t", res)
            assert_true(np.isclose(vv[1], res))

        # Invalid 'scaling'
        assert_raises(ValueError, around, 1234.567, 1, 'symlog', 'n')
        # Invalid 'dir'ection
        assert_raises(ValueError, around, 1234.567, 1, 'log', 'm')
        return

    def test_str_array(self):
        from zcode.math.math_core import str_array
        print("TestMathCore.test_str_array()")

        arr = np.linspace(0, 10.0, 6)
        sa = str_array(arr)
        print("'({})' ==> '{}'".format(arr, sa))
        assert_true(sa == '[0.00, 2.00, 4.00, 6.00, 8.00, 10.00]')

        sa = str_array(arr, (2, 2))
        print("'({}, (2, 2))' ==> '{}'".format(arr, sa))
        assert_true(sa == '[0.00, 2.00... 8.00, 10.00]')

        sa = str_array(arr, None)
        print("'({}, None)' ==> '{}'".format(arr, sa))
        assert_true(sa == '[0.00, 2.00, 4.00, 6.00, 8.00, 10.00]')

        sa = str_array(arr, 1)
        print("'({}, 1)' ==> '{}'".format(arr, sa))
        assert_true(sa == '[0.00... 10.00]')

        sa = str_array(arr, (1, 3))
        print("'({}, (1, 3))' ==> '{}'".format(arr, sa))
        assert_true(sa == '[0.00... 6.00, 8.00, 10.00]')

        sa = str_array(arr, (12, 10))
        print("'({}, (12, 10))' ==> '{}'".format(arr, sa))
        assert_true(sa == '[0.00, 2.00, 4.00, 6.00, 8.00, 10.00]')

        sa = str_array(arr, (2, 1), delim=' ')
        print("'({}, (2, 1), delim=' ')' ==> '{}'".format(arr, sa))
        assert_true(sa == '[0.00 2.00... 10.00]')

        sa = str_array(arr, (2, 1), format=':.1e')
        print("'({}, (2, 1), format=':.1e')' ==> '{}'".format(arr, sa))
        assert_true(sa == '[0.0e+00, 2.0e+00... 1.0e+01]')

        return


# Run all methods as if with `nosetests ...`
if __name__ == "__main__":
    run_module_suite()
