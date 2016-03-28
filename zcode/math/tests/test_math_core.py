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

    def test_confidenceBands(self):
        print("TestMathCore.test_confidenceBands")
        from zcode.math import confidenceBands
        np.random.seed(9865)
        uni1 = np.random.uniform(0.0, 1.0, size=100)
        uni2 = np.random.uniform(0.0, 1.0, size=100)
        zz = np.linspace(-np.pi, np.pi, num=20)
        yy = np.array([uni1 + np.sin(z) for z in zz])
        dz = np.average(np.diff(zz))
        xx = np.array([4.0*dz*(uni2-0.5) + z for z in zz])

        count, med, conf, xbins = confidenceBands(xx, yy, 20, 'lin', confInt=0.68)
        true_count = np.array([43,  68, 100, 115, 111, 114, 118, 121, 113, 110,
                               117, 112, 114, 120, 115, 114, 111,  95,  58,  31])
        true_med = np.array([0.44741627,  0.27070286,  0.13062732, -0.06400147, -0.30522202,
                             -0.35270415, -0.32772738, -0.21848981,  0.0700967,  0.38662969,
                             0.75191171,  1.07342266,  1.31406718,  1.43435843,  1.46189632,
                             1.31720551,  1.13144681,  0.92959401,  0.78686968,  0.67194954])
        true_conf = [[0.12095047, -0.17810892, -0.38115803, -0.51990728, -0.66618246,
                      -0.70931296, -0.6526213, -0.56719675, -0.30787827, -0.0927599,
                      0.22306699,  0.59704074,  0.93804672,  1.11844012,  1.05683467,
                      0.81939978,  0.57008734,  0.48195629,  0.35877874,  0.37992886],
                     [0.836671,  0.6102549,  0.48461273,  0.378781,  0.06250324,
                      -0.07095468, -0.04947349,  0.12198184,  0.53509063,  0.87365337,
                      1.18369298,  1.45002858,  1.62060513,  1.71902407,  1.76211033,
                      1.63896559,  1.49360737,  1.39381561,  1.17244189,  0.9467008]]
        true_xbins = np.array([-3.79679950e+00,  -3.41718685e+00,  -3.03757420e+00,
                               -2.65796156e+00,  -2.27834891e+00,  -1.89873626e+00,
                               -1.51912362e+00,  -1.13951097e+00,  -7.59898323e-01,
                               -3.80285676e-01,  -6.73029298e-04,   3.78939617e-01,
                               7.58552264e-01,   1.13816491e+00,   1.51777756e+00,
                               1.89739020e+00,   2.27700285e+00,   2.65661550e+00,
                               3.03622814e+00,   3.41584079e+00,   3.79545344e+00])

        assert_true(np.allclose(count, true_count))
        assert_true(np.allclose(med, true_med))
        assert_true(np.allclose(conf[:, 0], true_conf[0]))
        assert_true(np.allclose(conf[:, 1], true_conf[1]))
        assert_true(np.allclose(xbins, true_xbins))

        # plt.clf(); plt.scatter(xx, yy, color='b', alpha=0.5)
        # zcode.plot.plot_core.plotHistLine(plt.gca(), xbins, med, c='k', lw=2.0)
        # zcode.plot.plot_core.plotHistLine(plt.gca(), xbins, conf[:, 0], c='green', lw=2.0)
        # zcode.plot.plot_core.plotHistLine(plt.gca(), xbins, conf[:, 1], c='green', lw=2.0)
        # zcode.plot.plot_core.plotHistLine(plt.gca().twinx(), xbins, med, c='k', lw=2.0)

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
        return


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
        from zcode.math.math_core import _comparison_filter
        
        comp = ['g', '>']
        arr = [0.5, -1.0, 1.5, -0.5, 0.0]
        res = [0.5, 1.5]
        inds = [0, 2]
        for cc in comp:
            vals = _comparison_filter(arr, cc, value=0.0)
            assert_true(np.all(np.equal(vals, res)))
            vals = _comparison_filter(arr, cc, inds=True, value=0.0)
            assert_true(np.all(np.equal(vals[0], inds)))

        comp = ['le', '<=']
        arr = [0.5, -1.0, 1.5, -0.5, 0.0]
        res = [-1.0, -0.5, 0.0]
        inds = [1, 3, 4]
        for cc in comp:
            vals = _comparison_filter(arr, cc, value=0.0)
            assert_true(np.all(np.equal(vals, res)))
            vals = _comparison_filter(arr, cc, inds=True, value=0.0)
            assert_true(np.all(np.equal(vals[0], inds)))

        return


# Run all methods as if with `nosetests ...`
if __name__ == "__main__":
    run_module_suite()
