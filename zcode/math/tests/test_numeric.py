"""Test methods for `zcode/math/math_core.py`.

Can be run with:
    $ nosetests math/tests/test_math_core.py

"""

import numpy as np

# import zcodes
# import zcode.plot
from zcode.math import numeric


class Test_Numeric(object):

    @classmethod
    def setup_class(cls):
        np.random.seed(9865)
        cls.SIZE = 1000
        cls.r2 = np.random.uniform(-1.0, 1.0, size=cls.SIZE)

    def test_cumtrapz(self):

        bounds = [10.0, 23.0]
        # bounds = [10, 100]
        xx = np.logspace(0, 2, 100)
        # xx = np.logspace(0, 3, 91)

        for amp in [3.0, 3.25]:
            for gamma in [-1.8, 2.3]:
                print("\namp = {}, gamma = {}".format(amp, gamma))

                def func(zz):
                    return amp * np.power(zz, gamma)

                # y = dA/dx
                bounds = np.array(bounds)
                limits = func(bounds)
                exact = np.diff(limits * bounds)[0] / (gamma + 1)

                yy = func(xx)
                test_dadx = numeric.cumtrapz_loglog(yy, xx, bounds=bounds, dlogx=None)
                error = (test_dadx - exact) / exact
                print("dA/dx, true: {:.4e}, test: {:.4e}, error = {:.4e}".format(exact, test_dadx, error))
                assert_true(np.fabs(error) < 1e-6)

                # y = dA/dlog10x
                bounds = np.array(bounds)
                limits = func(bounds)
                exact = np.log(10.0) * np.diff(limits)[0] / gamma

                yy = func(xx)
                test_dadx = numeric.cumtrapz_loglog(yy, xx, bounds=bounds, dlogx=10.0)
                error = (test_dadx - exact) / exact
                print("dA/dlogx, true: {:.4e}, test: {:.4e}, error = {:.4e}".format(exact, test_dadx, error))
                assert_true(np.fabs(error) < 1e-6)

        return
