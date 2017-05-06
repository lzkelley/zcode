"""Test methods for `zcode/inout/timer.py`.

Can be run with:
    $ nosetests inout/tests/test_timer.py

"""
from __future__ import absolute_import, division, print_function, unicode_literals
from six.moves import xrange

from numpy.testing import run_module_suite
import numpy as np
# from nose.tools import assert_true, assert_false, assert_equal


class TestTimer(object):

    @classmethod
    def setup_class(cls):
        cls.NUM_ITER = 10
        pass

    @classmethod
    def teardown_class(cls):
        pass

    def test_timer(self):
        from zcode.inout.timer import Timings

        # Create `Timer` object
        times = Timings()

        for ii in xrange(self.NUM_ITER):
            times.start('one')
            np.random.randint(-1000, 1000, size=1000000)
            times.stop('one')

            times.start('two')
            NUM = 200
            ss = np.arange(3, NUM+1, 2)
            mroot = NUM ** 0.5
            half = (NUM + 1)//2 - 1
            ii = 0
            mm = 3
            while mm <= mroot:
                if ss[ii]:
                    jj = np.int((mm * mm - 3)/2)
                    ss[jj] = 0
                    while jj < half:
                        ss[jj] = 0
                        jj += mm
                ii += 1
                mm = 2*ii + 3

            times.stop('two')

            times.start('three')
            np.sort(np.random.permutation(np.arange(1000000)))
            times.stop('three')

        # for ii in xrange(len(times)):
        #     names = times.names()
        #     print(names[ii])
        #     for jj in times.durations[ii]:
        #         print(jj, end=' ')
        #     print("\n")
        #
        # print("Averages = ", times.average())

        times.report()


# Run all methods as if with `nosetests ...`
if __name__ == "__main__":
    run_module_suite()
