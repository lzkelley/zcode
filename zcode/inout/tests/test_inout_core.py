"""Test methods for `inout_core.py`.

Can be run with: 
    $ nosetests zcode/inout/tests/test_inout_core.py

"""
from __future__ import absolute_import, division, print_function, unicode_literals

import os
from numpy.testing import run_module_suite
import numpy as np
from nose.tools import assert_true, assert_false, assert_equal, assert_raises

# from zcode.inout import inout_core


class TestInoutCore(object):

    @classmethod
    def setup_class(cls):
        cls.fname_npz = '_test_inout_core_testfile.npz'
        cls.fname_npz_subdir = os.path.join('./subdir', cls.fname_npz)

    @classmethod
    def teardown_class(cls):
        pass
            
    def test_dictToNPZ_npzToDict(self):
        fname = self.fname_npz
        fname_subdir = self.fname_npz_subdir
        from zcode.inout.inout_core import npzToDict, dictToNPZ
        
        # Create a test dictionary to save
        subdata = {'a':'a', 'b':'abc', 'c':np.arange(4)}
        data = {'one':np.array(1), 'two':np.array(2, dtype=np.uint64), 'three':subdata}

        # Try saving
        dictToNPZ(data, fname)
        assert_true(os.path.exists(fname))

        # Try Loading
        loaded = npzToDict(fname)
        for key, item in data.items():
            print("key = ", key)
            print("\t", type(loaded[key]), repr(loaded[key]))
            print("\t", type(item), repr(item))
            # Look at internal dictionary separately
            if type(item) is not dict and type(loaded[key]) is not dict:
                assert_true(np.array_equal(loaded[key], item))
                assert_equal(type(loaded[key]), type(item))

        # Check internal dictionary
        subloaded = loaded['three']
        print("Subloaded keys = ", subloaded.keys())
        for key, item in subdata.items():
            print("key = ", key)
            print("\t", subloaded[key])
            print("\t", item)
            assert_true(np.array_equal(subloaded[key], item))
            assert_equal(type(subloaded[key]), type(item))

        # Delete temp file
        if os.path.exists(fname):
            os.remove(fname)
        assert_false(os.path.exists(fname))
        
        # Make sure subdirectories are created if needed
        dictToNPZ(data, fname_subdir)
        assert_true(os.path.exists(fname_subdir))

        # Delete temp file
        if os.path.exists(fname_subdir):
            os.remove(fname_subdir)
        assert_false(os.path.exists(fname_subdir))



# Run all methods as if with `nosetests ...`
if __name__ == "__main__":
    run_module_suite()
