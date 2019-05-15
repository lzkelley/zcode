"""Test methods for `zcode/math/math_core.py`.

Can be run with:
    $ nosetests math/tests/test_math_core.py

"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import scipy as sp
from numpy.testing import run_module_suite
from nose.tools import assert_true, assert_false

import zcode.math as zmath


class Test_KDE(object):

    @classmethod
    def setup_class(cls):
        np.random.seed(9865)

    def test_compare_scipy_1d(self):
        print("\Test_KDE:test_compare_scipy_1d()")
        NUM = 100
        a1 = np.random.normal(6.0, 1.0, NUM//2)
        a2 = np.random.lognormal(0, 0.5, size=NUM//2)
        aa = np.concatenate([a1, a2])

        bins = zmath.spacing([-1, 14.0], 'lin', 40)
        grid = zmath.spacing(bins, 'lin', 3000)

        methods = ['scott', 0.04, 0.2, 0.8]
        classes = [sp.stats.gaussian_kde, zmath.kde.gaussian_kde]
        for mm in methods:
            kdes = [cc(aa, bw_method=mm).pdf(grid) for cc in classes]
            print("bw_method='{}'".format(mm))
            assert_true(np.allclose(kdes[0], kdes[1]))

        return

    def test_compare_scipy_2d(self):
        print("\Test_KDE:test_compare_scipy_2d()")

        NUM = 1000
        a1 = np.random.normal(6.0, 1.0, NUM//2)
        a2 = np.random.lognormal(0, 0.5, size=NUM//2)
        aa = np.concatenate([a1, a2])

        bb = np.random.normal(3.0, 0.02, NUM) + aa/100

        data = [aa, bb]
        edges = [zmath.spacing(dd, 'lin', 30, stretch=0.5) for dd in data]
        cents = [zmath.midpoints(ee, 'lin') for ee in edges]

        xe, ye = np.meshgrid(*edges)
        xc, yc = np.meshgrid(*cents)
        grid = np.vstack([xc.ravel(), yc.ravel()])

        methods = ['scott', 0.04, 0.2, 0.8]
        classes = [sp.stats.gaussian_kde, zmath.kde.gaussian_kde]
        for mm in methods:
            print("bw_method='{}'".format(mm))
            kdes = [cc(data, bw_method=mm).pdf(grid).reshape(xc.shape).T for cc in classes]
            assert_true(np.allclose(kdes[0], kdes[1]))

        return


# Run all methods as if with `nosetests ...`
if __name__ == "__main__":
    run_module_suite()
