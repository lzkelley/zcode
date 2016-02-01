"""Test methods for `zcode/inout/singleton.py`.

Can be run with:
    $ nosetests inout/tests/test_singleton.py
from the root `zcode` directory (i.e. directory with 'README.rst').

"""
from __future__ import absolute_import, division, print_function, unicode_literals
# from six.moves import xrange

from numpy.testing import run_module_suite
from nose.tools import assert_true, assert_equal


class TestSingleton(object):

    @classmethod
    def setup_class(cls):
        pass

    @classmethod
    def teardown_class(cls):
        pass

    def test_singleton(self):
        from zcode.inout.singleton import Singleton

        # Create first instance
        one = Singleton()
        one.a = 1.0
        one.b = 'b'
        # Create a second instance
        two = Singleton()
        two.a = 2.0
        two.b = 'c'
        two.c = 'test'

        # Make sure parameters which are 'changed' are equal across both objects
        assert_equal(one.a, two.a)
        assert_equal(one.b, two.b)
        # Make sure parameter added to one instance exists on the other
        assert_true(hasattr(one, 'c'))

        for ii, obj in enumerate([one, two]):
            print("Object {}: a = '{}', b = '{}'\n\t__dict__ = '{}'".format(
                ii, obj.a, obj.b, obj.__dict__))

        # raise ValueError()

    def test_singleton_inherit(self):
        from zcode.inout.singleton import Singleton

        # Declare derived class
        class sing(Singleton):
            def __init__(self):
                self.one = 1
                self.two = 2.0
                self.three = 'c'

        # Create instances
        aa = sing()
        aa.test = 'test'
        bb = sing()
        bb.one = 11
        bb.two = 22.2

        print("aa = ", aa.__dict__)
        print("bb = ", bb.__dict__)

        for ii, obj in enumerate([aa, bb]):
            print("Object {}: one = '{}', two = '{}', three = '{}'\n\t__dict__ = '{}'".format(
                ii, obj.one, obj.two, obj.three, obj.__dict__))

        raise ValueError()

# Run all methods as if with `nosetests ...`
if __name__ == "__main__":
    run_module_suite()
