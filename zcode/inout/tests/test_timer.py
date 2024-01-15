"""Test methods for `zcode/inout/timer.py`.

Can be run with:
    $ nosetests inout/tests/test_timer.py

"""

import numpy as np


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

        for ii in range(self.NUM_ITER):
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
                    jj = int((mm * mm - 3)/2)
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

        times.report()

