"""Test methods for `zcode/plot/Hist2D.py`.

Can be run with:
    $ nosetests plot/tests/test_hist2d.py

"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from numpy.testing import run_module_suite
from nose.tools import assert_true

from zcode.plot import Hist2D
import zcode.inout as zio


def save_fig(fig, fname, count):
    save_name = zio.modify_filename(fname, append="_{}".format(count))
    fig.savefig(fig, save_name)
    print("Saved {} to '{}'".format(count, save_name))
    return count+1


class TestHist2D(object):

    @classmethod
    def setup_class(cls):
        NUM = 1000
        cls.NUM = NUM
        cls.xx = np.random.normal(1.0, 0.5, size=NUM) + np.random.uniform(size=NUM)
        cls.yy = cls.xx*np.exp(-(cls.xx-1.0)**2)
        cls.yy += np.random.normal(0.0, 0.5, size=NUM) + np.random.uniform(size=NUM)
        cls.zz = np.exp(-(cls.xx - 1.5)**2 - (cls.yy - 1.0)**2)
        pass

    def test_plot2DHistProj(self):
        print("TestHist2D.test_plot2DHistProj")
        '''
        from zcode.plot.Hist2D import plot2DHistProj
        fname = "TestHist2D_test_plot2DHistProj.pdf"
        count = 0

        xx = self.xx
        yy = self.yy
        zz = self.zz
        NUM = self.NUM

        tt = np.ones(NUM)
        print(np.shape(xx), np.shape(yy), np.shape(tt))
        fig = plot2DHistProj(xx, yy, weights=tt)
        count = save_fig(fig, fname, count)
        '''
        return


# Run all methods as if with `nosetests ...`
if __name__ == "__main__":
    run_module_suite()
