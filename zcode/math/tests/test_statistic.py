"""Test methods for `zcode/math/math_core.py`.

Can be run with:
    $ nosetests math/tests/test_math_core.py

"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from numpy.testing import run_module_suite
from nose.tools import assert_true


class TestStatistic(object):

    @classmethod
    def setup_class(cls):
        np.random.seed(9865)

    def test_confidence_bands(self):
        print("TestMathCore.test_confidence_bands")
        from zcode.math import statistic
        np.random.seed(9865)
        uni1 = np.random.uniform(0.0, 1.0, size=100)
        uni2 = np.random.uniform(0.0, 1.0, size=100)
        zz = np.linspace(-np.pi, np.pi, num=20)
        yy = np.array([uni1 + np.sin(z) for z in zz])
        dz = np.average(np.diff(zz))
        xx = np.array([4.0*dz*(uni2-0.5) + z for z in zz])

        count, med, conf, xbins = statistic.confidence_bands(xx, yy, 20, 'lin', confInt=0.68)
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
        return

    def test_sigma(self):
        from zcode.math import statistic
        sigma = [1.0, 2.0]
        inside = [0.68268949213708585, 0.95449973610364158]
        outside = 1 - np.array(inside)

        # Areas
        ret_inside = statistic.sigma(sigma, side='in')
        ret_outside = statistic.sigma(sigma, side='out')
        print("sigma = ", sigma)
        print("inside = ", inside)
        print("ret_inside = ", ret_inside)
        print("outside = ", outside)
        print("ret_outside = ", ret_outside)
        assert_true(np.allclose(inside, ret_inside))
        assert_true(np.allclose(outside, ret_outside))

        # Boundaries
        sig_1 = 1.0
        sig_2 = 2.0
        inside_1 = [0.158655253931, 0.841344746069]
        outside_2 = [0.0227501319482, 0.977249868052]

        ret_inside = statistic.sigma(sig_1, side='in', boundaries=True)
        ret_outside = statistic.sigma(sig_2, side='out', boundaries=True)
        print("sigma_1 = ", sig_1)
        print("inside_1 = ", inside_1)
        print("ret_inside = ", ret_inside)
        print("sigma_2 = ", sig_2)
        print("outside_2 = ", outside_2)
        print("ret_outside = ", ret_outside)
        assert_true(np.allclose(inside_1, ret_inside))
        assert_true(np.allclose(outside_2, ret_outside))
        return


# Run all methods as if with `nosetests ...`
if __name__ == "__main__":
    run_module_suite()
