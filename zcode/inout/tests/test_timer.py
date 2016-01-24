"""Test methods for `zcode/inout/timer.py`.

Can be run with:
    $ nosetests inout/tests/test_timer.py

"""
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import warnings
import shutil
from numpy.testing import run_module_suite
import numpy as np
from nose.tools import assert_true, assert_false, assert_equal


class TestTimer(object):

    @classmethod
    def setup_class(cls):
        pass

    @classmethod
    def teardown_class(cls):
        pass

    def test_dictToNPZ_npzToDict(self):
        from zcode.inout.timer import Timer

        times = Timer()


# Run all methods as if with `nosetests ...`
if __name__ == "__main__":
    run_module_suite()
