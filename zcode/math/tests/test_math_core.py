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
import scipy.stats  # noqa
from nose.tools import assert_true, assert_false, assert_equal, assert_raises, assert_almost_equal

from zcode.math import math_core, interpolate


class TestMathCore(object):

    @classmethod
    def setup_class(cls):
        np.random.seed(9865)
        cls.SIZE = 100
        cls.r1 = np.random.random(cls.SIZE)
        cls.r2 = np.random.uniform(-1.0, 1.0, size=cls.SIZE)

    def test_argnearest_ordered(self):
        from zcode.math.math_core import argnearest
        edges = np.array([0.2, 0.8, 1.3, 1.5, 2.0, 3.1, 3.8, 3.9, 4.5, 5.1, 5.5])
        vals = np.array([-1, 0.2, 1, 1.4, 2, 3, 4, 5, 5.5, 10])
        correct = [0, 0, 1, 2, 4, 5, 7, 9, 10, 10]
        retval = argnearest(edges, vals, assume_sorted=True)
        assert_true(np.all(correct == retval))
        print("Edges = {}".format(edges))
        print("Vals = {}".format(vals))
        print("retval  = {}".format(retval))
        print("correct = {}".format(correct))
        return

    def test_argnearest_ordered_left_right(self):
        from zcode.math.math_core import argnearest
        #                 0    1    2    3    4    5    6    7    8    9    10
        edges = np.array([0.2, 0.8, 1.3, 1.5, 2.0, 3.1, 3.8, 3.9, 4.5, 5.1, 5.5])
        vals    = np.array([-1, 0.2, 1, 1.4, 2, 3, 4, 5, 5.5, 10])
        correct = np.array([-1, -1 , 1, 2  , 3, 4, 7, 8, 9  , 10])
        print("LEFT")
        retval = argnearest(edges, vals, assume_sorted=True, side='left')
        print("Edges = {}".format(edges))
        print("Vals = {}".format(vals))
        print("retval  = {}".format(retval))
        print("correct = {}".format(correct))
        print(correct == retval)
        print(np.all(correct == retval))
        assert_true(np.all(correct == retval))

        correct += 1
        for ee in edges:
            correct[vals == ee] += 1

        print("RIGHT")
        retval = argnearest(edges, vals, assume_sorted=True, side='right')
        print("Edges = {}".format(edges))
        print("Vals = {}".format(vals))
        print("retval  = {}".format(retval))
        print("correct = {}".format(correct))
        assert_true(np.all(correct == retval))

        return

    def test_argnearest_unordered_x(self):
        from zcode.math.math_core import argnearest
        edges = np.array([0.2, 0.8, 1.3, 1.5, 2.0, 3.1, 3.8, 3.9, 4.5, 5.1, 5.5])
        vals = np.array([-1, 0.2, 1, 1.4, 2, 3, 4, 5, 5.5, 10])
        correct = np.array([2, 2, 8, 7, 0, 6, 9, 3, 4, 4])

        # ix = np.random.permutation(edges.size)
        ix = np.array([4,  3,  0,  9, 10,  8,  5,  2,  1,  7,  6])
        edges = edges[ix]

        retval = argnearest(edges, vals, assume_sorted=False)
        print("Edges = {}".format(edges))
        print("Vals = {}".format(vals))
        print("retval  = {}".format(retval))
        print("nearest = {}".format(edges[retval]))
        print("Vals = {}".format(vals))
        print("retval  = {}".format(retval))
        print("correct = {}".format(correct))
        assert_true(np.all(correct == retval))
        return

    def test_argnearest_unordered_xy(self):
        from zcode.math.math_core import argnearest
        edges = np.array([0.2, 0.8, 1.3, 1.5, 2.0, 3.1, 3.8, 3.9, 4.5, 5.1, 5.5])
        vals = np.array([-1, 0.2, 1, 1.4, 2, 3, 4, 5, 5.5, 10])
        correct = np.array([0, 7, 6, 3, 4, 9, 2, 4, 2, 8])

        # ix = np.random.permutation(edges.size)
        ix = np.array([4,  3,  0,  9, 10,  8,  5,  2,  1,  7,  6])
        edges = edges[ix]
        iy = np.array([4, 3, 5, 7, 9, 6, 0, 8, 1, 2])
        vals = vals[iy]

        retval = argnearest(edges, vals, assume_sorted=False)
        print("Edges = {}".format(edges))
        print("Vals = {}".format(vals))
        print("retval  = {}".format(retval))
        print("nearest = {}".format(edges[retval]))
        print("Vals = {}".format(vals))
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

        arr = np.array(arr)
        for cc in comp:
            vals = comparison_filter(arr, cc, value=0.0)
            assert_true(np.all(np.equal(vals, res)))
            val_inds = comparison_filter(arr, cc, inds=True, value=0.0)
            assert_true(np.all(np.equal(arr[val_inds], arr[inds])))

        comp = ['le', '<=']
        arr = [0.5, -1.0, 1.5, -0.5, 0.0]
        res = [-1.0, -0.5, 0.0]
        inds = [1, 3, 4]

        arr = np.array(arr)
        for cc in comp:
            vals = comparison_filter(arr, cc, value=0.0)
            assert_true(np.all(np.equal(vals, res)))
            vals = comparison_filter(arr, cc, inds=True, value=0.0)
            assert_true(np.all(np.equal(arr[vals], arr[inds])))

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
        correct = '[0.00, 2.00, 4.00, 6.00, 8.00, 10.00]'
        sa = str_array(arr)
        print("'({})' ==> '{}', should be '{}'".format(arr, sa, correct))
        assert_true(sa == correct)

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

    def test_broadcast(self):
        from zcode.math.math_core import broadcast

        def check_in_ot(din, check):
            dot = broadcast(*din)
            print("input:  {}".format(din))
            print("output: {} ({})".format(dot, check))
            assert_true(np.all([dd == cc for dd, cc in zip(dot, check)]))
            assert_true(np.all([np.shape(dd) == np.shape(cc) for dd, cc in zip(dot, check)]))
            return

        # Normal broadcast (1,) (2,) ==> (2,) (2,)
        din = [[1.0], [2.0, 3.0]]
        check = [[[1.0, 1.0]], [[2.0, 3.0]]]
        check_in_ot(din, check)

        # Scalar-only broadcast () () ==> () ()
        din = [1.0, 2.0]
        check = din
        check_in_ot(din, check)

        # Mixed scalar and array
        din = [1.5, [1.0, 2.0], [1.0, 2.0, 3.0]]
        check = [
            [[1.5, 1.5, 1.5], [1.5, 1.5, 1.5]],
            [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
            [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]
        ]
        check_in_ot(din, check)

        din = [[1.0], [2.0, 3.0]]
        check = [[[1.0, 1.0]], [[2.0, 3.0]]]
        dot = broadcast(*din)
        check_in_ot(din, check)

        sh_in = np.random.randint(1, 5, 3)
        sh_ot = [sh_in for ii in range(len(sh_in))]
        din = [np.random.normal(size=sh) for sh in sh_in]
        dot = broadcast(*din)
        print("Input shapes: '{}'".format(sh_in))
        print("Output shapes: '{}' ({})".format([dd.shape for dd in dot], sh_ot))
        assert_true(np.all([dd.shape == sh for dd, sh in zip(dot, sh_ot)]))
        return


class Test_Interp(object):

    def test_interp_lin_lin(self):
        print("\ntest_interp_lin_lin()")
        kw = dict(xlog=False, ylog=False, valid=False, left=np.nan, right=100.0)

        xo = [1.0, 2.0, 3.0]
        yo = [10.0, 20.0, 30.0]

        tests = [1.5, 2.5, 0.5, 3.5]
        truth = [15.0, 25.0, np.nan, 100.0]

        for xx, zz in zip(tests, truth):
            yy = interpolate.interp(xx, xo, yo, **kw)
            print("{} ==> {}, should be {}".format(xx, yy, zz))
            if np.isnan(zz):
                assert_true(np.isnan(yy))
            else:
                assert_almost_equal(yy, zz)

        return

    def test_interp_lin_log(self):
        print("\ntest_interp_lin_log()")
        kw = dict(xlog=False, ylog=True, valid=False, left=100.0, right=np.nan)

        xo = [1.0, 2.0, 3.0]
        yo = [1.0e1, 1.0e3, 1.0e5]

        tests = [1.5, 2.5, 0.5, 3.5]
        truth = [1.0e2, 1.0e4, 100.0, np.nan]

        for xx, zz in zip(tests, truth):
            yy = interpolate.interp(xx, xo, yo, **kw)
            print("{} ==> {}, should be {}".format(xx, yy, zz))
            if np.isnan(zz):
                assert_true(np.isnan(yy))
            else:
                assert_almost_equal(yy, zz)

        return

    def test_interp_log_lin(self):
        print("\ntest_interp_log_lin()")
        kw = dict(xlog=True, ylog=False, valid=False, left=100.0, right=np.nan)

        xo = [2.0e-5, 2.0e-3, 2.0e-1]
        yo = [-10.0, -20.0, -30.0]

        tests = [2.0e-4, 2.0e-2, 1.0e-8, 1.0e8]
        truth = [-15.0, -25.0, 100.0, np.nan]

        for xx, zz in zip(tests, truth):
            yy = interpolate.interp(xx, xo, yo, **kw)
            print("{} ==> {}, should be {}".format(xx, yy, zz))
            if np.isnan(zz):
                assert_true(np.isnan(yy))
            else:
                assert_almost_equal(yy, zz)

        return

    def test_interp_log_log(self):
        print("\ntest_interp_log_log()")
        kw = dict(xlog=True, ylog=True, valid=False, left=np.nan, right=100.0)

        xo = [1.0e-1, 1.0e1, 1.0e5]
        yo = [3.0e0, 3.0e-2, 3.0e6]

        tests = [1.0, 1.0e3, 1.0e-8, 1.0e8]
        truth = [3.0e-1, 3.0e2, np.nan, 100.0]

        for xx, zz in zip(tests, truth):
            yy = interpolate.interp(xx, xo, yo, **kw)
            print("{} ==> {}, should be {}".format(xx, yy, zz))
            if np.isnan(zz):
                assert_true(np.isnan(yy))
            else:
                assert_almost_equal(yy, zz)

        return


class Test_Interp_Func_Linear(object):

    KW = dict(kind='linear', bounds_error=False)

    def test_interp_func(self):
        print("\n|test_interp_func()|")
        options = [True, False]
        TRIES = 10
        SAMPS = 40
        TESTS = 100
        LOG_RANGE = [-8.0, 8.0]

        for xlog in options:
            for ylog in options:
                kw = dict(xlog=xlog, ylog=ylog)
                print("xlog = {}, ylog = {}".format(xlog, ylog))

                for kk in range(TRIES):
                    xo = np.random.uniform(*LOG_RANGE, SAMPS)
                    xo = np.sort(xo)
                    yo = np.random.uniform(*LOG_RANGE, SAMPS)

                    xx = np.random.uniform(*math_core.minmax(xo), TESTS)

                    if xlog:
                        xo = np.power(10.0, xo)
                        xx = np.power(10.0, xx)
                    if ylog:
                        yo = np.power(10.0, yo)

                    y1 = interpolate.interp(xx, xo, yo, valid=False, **kw)
                    y2 = interpolate.interp_func(xo, yo, kind='linear', bounds_error=False, **kw)(xx)
                    assert_true(np.allclose(y1, y2))

        return

    def test_interp_func_lin_lin(self):
        print("\n|test_interp_func_lin_lin()|")
        kw = dict(xlog=False, ylog=False, fill_value=(np.nan, 100.0))
        kw.update(self.KW)

        xo = [1.0, 2.0, 3.0]
        yo = [10.0, 20.0, 30.0]

        tests = [1.5, 2.5, 0.5, 3.5]
        truth = [15.0, 25.0, np.nan, 100.0]

        for xx, zz in zip(tests, truth):
            yy = interpolate.interp_func(xo, yo, **kw)(xx)
            print("{} ==> {}, should be {}".format(xx, yy, zz))
            if np.isnan(zz):
                assert_true(np.isnan(yy))
            else:
                assert_almost_equal(yy, zz)

        return

    def test_interp_func_lin_log(self):
        print("\n|test_interp_func_lin_log()|")
        kw = dict(xlog=False, ylog=True, fill_value=(100.0, np.nan))
        kw.update(self.KW)

        xo = [1.0, 2.0, 3.0]
        yo = [1.0e1, 1.0e3, 1.0e5]

        tests = [1.5, 2.5, 0.5, 3.5]
        truth = [1.0e2, 1.0e4, 100.0, np.nan]

        for xx, zz in zip(tests, truth):
            yy = interpolate.interp_func(xo, yo, **kw)(xx)
            print("{} ==> {}, should be {}".format(xx, yy, zz))
            if np.isnan(zz):
                assert_true(np.isnan(yy))
            else:
                assert_almost_equal(yy, zz)

        return

    def test_interp_func_log_lin(self):
        print("\n|test_interp_func_log_lin()|")
        kw = dict(xlog=True, ylog=False, fill_value=(100.0, np.nan))
        kw.update(self.KW)

        xo = [2.0e-5, 2.0e-3, 2.0e-1]
        yo = [-10.0, -20.0, -30.0]

        tests = [2.0e-4, 2.0e-2, 1.0e-8, 1.0e8]
        truth = [-15.0, -25.0, 100.0, np.nan]

        for xx, zz in zip(tests, truth):
            yy = interpolate.interp_func(xo, yo, **kw)(xx)
            print("{} ==> {}, should be {}".format(xx, yy, zz))
            if np.isnan(zz):
                assert_true(np.isnan(yy))
            else:
                assert_almost_equal(yy, zz)

        return

    def test_interp_func_log_log(self):
        print("\n|test_interp_func_log_log()|")
        kw = dict(xlog=True, ylog=True, fill_value=(np.nan, 100.0))
        kw.update(self.KW)

        xo = [1.0e-1, 1.0e1, 1.0e5]
        yo = [3.0e0, 3.0e-2, 3.0e6]

        tests = [1.0, 1.0e3, 1.0e-8, 1.0e8]
        truth = [3.0e-1, 3.0e2, np.nan, 100.0]

        for xx, zz in zip(tests, truth):
            yy = interpolate.interp_func(xo, yo, **kw)(xx)
            print("{} ==> {}, should be {}".format(xx, yy, zz))
            if np.isnan(zz):
                assert_true(np.isnan(yy))
            else:
                assert_almost_equal(yy, zz)

        return


class Test_Interp_Func_Mono(object):

    KW = dict(kind='mono')

    def test_interp_func(self):
        print("\n|test_interp_func()|")

        xo = [0.1, 1.0, 2.0, 3.0, 4.0, 5.0]
        yo = [100.0, 100.0, 90.0, 0.1, 2.0, 2.0]
        NUM = len(xo)

        xn = np.linspace(xo[1], xo[-2], 1000)

        def test_within(xx, yy):
            vals = []
            for ii in range(NUM-1):
                xl = xo[ii]
                xh = xo[ii+1]
                yl = yo[ii]
                yh = yo[ii+1]

                inds = (xl <= xx) & (xx <= xh)
                rv1 = math_core.within(yy[inds], [yl, yh], all=True, close=True)
                rv2 = math_core.mono(yy[inds], 'ge') or math_core.mono(yy[inds], 'le')
                rv = (rv1 and rv2)
                vals.append(rv)

            return np.all(vals)

        options = [True, False]
        for xlog in options:
            for ylog in options:
                func = interpolate.interp_func(xo, yo, xlog=xlog, ylog=ylog, kind='mono')
                yn = func(xn)
                print("xlog = {}, ylog = {}".format(xlog, ylog))
                assert_true(test_within(xn, yn))

                # 'cubic' should be NON-monotonic, make sure test shows that
                func = interpolate.interp_func(xo, yo, xlog=xlog, ylog=ylog, kind='cubic')
                yn = func(xn)
                assert_false(test_within(xn, yn))

        return


class Test_Edges_From_Cents(object):

    def test_lin_spacing(self):
        print("\n|test_lin_spacing()|")

        edges_true = [
            np.linspace(0.0, 1.0, 20),
            np.linspace(1.0, 0.0, 20),
            np.linspace(-100, 100, 100)
        ]

        for true in edges_true:
            cents = math_core.midpoints(true, log=False)
            edges = math_core.edges_from_cents(cents, log=False)

            print("truth = {}".format(math_core.str_array(true)))
            print("recov = {}".format(math_core.str_array(edges)))
            assert_true(np.allclose(edges, true))

        return

    def test_log_spacing(self):
        print("\n|test_log_spacing()|")

        true_pars = [
            [0.0, 1.0, 20],
            [1.0, 0.0, 20],
            [2.0, -2.0, 100]
        ]

        for pars in true_pars:
            true = np.logspace(*pars)
            cents = math_core.midpoints(true, log=True)
            edges = math_core.edges_from_cents(cents, log=True)
            print("pars = ", pars)
            print("truth = {}".format(math_core.str_array(true)))
            print("recov = {}".format(math_core.str_array(edges)))
            assert_true(np.allclose(edges, true))

        return

    def test_irr_spacing(self):
        print("\n|test_irr_spacing()|")

        NUM = 10
        xx = np.arange(NUM)
        widths = 1.5 + 0.4*xx + 0.1*(xx**2)

        true = np.zeros(NUM+1)
        true[0] = 4.0
        for ii in range(1, NUM+1):
            true[ii] = true[ii-1] + widths[ii-1]

        cents = math_core.midpoints(true, log=False)
        edges = math_core.edges_from_cents(cents, log=False)
        print("truth = {}".format(math_core.str_array(true)))
        print("recov = {}".format(math_core.str_array(edges)))
        assert_true(np.allclose(edges, true, rtol=1e-1))

        return


# Run all methods as if with `nosetests ...`
if __name__ == "__main__":
    run_module_suite()
