"""Test methods for `zcode/math/math_core.py`.

"""

import numpy as np

from zcode.math import interpolate


class Test_Interp_Axis(object):

    @classmethod
    def setup_class(cls):
        np.random.seed(9865)
        return

    def test(self):
        print("Test_Interp_Axis")
        print(f"\tTest_Interp_Axis.test()")
        for ndim in range(1, 4):
            self._test_ndim(ndim)
        return

    def _test_ndim(self, ndim):
        print("Test_Interp_Axis")
        print(f"\tTest_Interp_Axis._test_ndim() ndim = {ndim}")

        for axis in range(ndim):
            shape = np.random.randint(3, 10, ndim)
            print(axis, shape)

            xx = np.random.uniform(-100, 100, shape)
            yy = np.random.uniform(-100, 100, shape)

            for xnew in [0.5, [-2.5, 1.5]]:

                test = interpolate.interp_axis(xnew, xx, yy, axis=axis, xlog=False, ylog=False)

                xx = np.moveaxis(xx, axis, -1)
                yy = np.moveaxis(yy, axis, -1)
                sh = xx.shape[:-1]
                if np.isscalar(xnew):
                    check = np.zeros(sh)
                elif len(xnew) == 2:
                    check = np.zeros(sh + (2,))
                else:
                    raise ValueError()

                for idx in np.ndindex(sh):
                    aa = xx[idx]
                    bb = yy[idx]
                    ss = np.argsort(aa)
                    zz = np.interp(xnew, aa[ss], bb[ss], left=np.nan, right=np.nan)
                    check[idx] = zz

                if not np.all(np.isfinite(test) == np.isfinite(check)):
                    raise ValueError("Finite check failed!")

                idx = np.isfinite(test)
                if not np.allclose(test[idx], check[idx]):
                    raise ValueError("Match check failed!")

        return
