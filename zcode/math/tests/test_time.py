"""Test methods for `zcode/math/time.py`.

Can be run with:
    $ nosetests math/tests/test_time.py

"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import astropy as ap
import astropy.time
import datetime

from numpy.testing import run_module_suite
from nose.tools import assert_true

import zcode
import zcode.math.time


class TestTime(object):

    @classmethod
    def setup_class(cls):
        np.random.seed(9865)

    def test_to_datetime(self):
        print("TestTime.test_to_datetime")

        # These should all work correctly
        #    Each row is [description, input, correct-output]
        test_cases = [
            ["unix", 1491405804.00, (2017, 4, 5, 15, 23, 24)],
            ["date & time", "2017-05-04 15:23:24.0124", (2017, 5, 4, 15, 23, 24, 12400)],
            ["date", "2017-05-04", (2017, 5, 4)],

            ["astropy time", ap.time.Time("2017-05-04 15:23:24.0124"),
             (2017, 5, 4, 15, 23, 24, 12400)],
            ["datetime", datetime.datetime(2017, 5, 4, 15, 23, 24, 12400),
             (2017, 5, 4, 15, 23, 24, 12400)],

            ["complex", "Fri May 4th 15:23:24 2017", (2017, 5, 4, 15, 23, 24)]
        ]

        for test in test_cases:
            correct = datetime.datetime(*test[2])
            print("Type: {} ({}), input: {}".format(test[0], type(test[1]), test[1]))
            print("\tcorrect = '{}'".format(correct))
            retval = zcode.math.time.to_datetime(test[1])
            print("\tretval  = '{}' ({})".format(retval, type(retval)))
            assert_true(retval == correct)

        # Try using the `format` argument
        correct = datetime.datetime(2017, 5, 4, 15, 23, 24, 12400)
        form = "%Y|%j|%H...%S.%f&%M"
        weird = correct.strftime(form)
        print("weird format: '{}', '{}'".format(weird, form))
        print("\tcorrect = '{}'".format(correct))
        retval = zcode.math.time.to_datetime(weird, format=form)
        print("\tretval  = '{}'".format(retval))
        assert_true(retval == correct)

        return


# Run all methods as if with `nosetests ...`
if __name__ == "__main__":
    run_module_suite()
