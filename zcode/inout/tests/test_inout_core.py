"""Test methods for `inout_core.py`.

Can be run with:
    $ nosetests zcode/inout/tests/test_inout_core.py

"""
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import warnings
from numpy.testing import run_module_suite
import numpy as np
from nose.tools import assert_true, assert_false, assert_equal, assert_raises


class TestInoutCore(object):

    @classmethod
    def setup_class(cls):
        cls.fname_npz = '_test_inout_core_testfile.npz'
        cls.fname_npz_subdir = os.path.join('./subdir', cls.fname_npz)
        cls.test_dir_0 = '_test_inout_core_dir'
        cls.test_file_0 = '_test_filename.txt'

    @classmethod
    def teardown_class(cls):
        pass

    def test_dictToNPZ_npzToDict(self):
        fname = self.fname_npz
        fname_subdir = self.fname_npz_subdir
        from zcode.inout.inout_core import npzToDict, dictToNPZ

        # Create a test dictionary to save
        subdata = {'a': 'a', 'b': 'abc', 'c': np.arange(4)}
        data = {'one': np.array(1), 'two': np.array(2, dtype=np.uint64), 'three': subdata}

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
            os.rmdir(os.path.dirname(fname_subdir))
        assert_false(os.path.exists(fname_subdir))

    def test_modify_exists(self):
        fdir = self.test_dir_0
        fname = self.test_file_0
        num_files = 4
        max_files = 20   # This must be between [11, 100]
        self.rem_files = []
        from zcode.inout.inout_core import modify_exists, modifyFilename

        # Create test directory if needed, store boolean whether to later remove it.
        kill_dir = False
        if not os.path.exists(fdir):
            os.makedirs(fdir)
            kill_dir = True

        # Create test filename
        fname = os.path.join(fdir, fname)
        # Make sure it doesnt already exist
        if os.path.exists(fname):
            raise RuntimeError("Test filename '{}' already exists.".format(fname))

        # Create files that should *not* interfere with 'modify_exists'
        #    'modify_exists' should only look for 2-digit appended numbers
        fname_distract_1 = modifyFilename(fname, append='_6')
        fname_distract_2 = modifyFilename(fname, append='_123')
        print("Interference filenames = '{}', '{}'".format(fname_distract_1, fname_distract_2))
        for ff in [fname_distract_1, fname_distract_2]:
            open(ff, 'a')
            rem_files.append(ff)

        # Test that filenames are appropriately modified
        # ----------------------------------------------
        print("fname = '{}'".format(fname))
        for ii in range(num_files):
            newName = modify_exists(fname, max=max_files)
            print(ii, "newName = ", newName)
            assert_false(os.path.exists(newName))
            # Create file
            open(newName, 'a')
            if ii == 0:
                intended_name = str(fname)
            else:
                intended_name = modifyFilename(fname, append="_{:02d}".format(ii-1))

            print("\tshould be = ", intended_name)
            assert_true(os.path.exists(intended_name))
            rem_files.append(newName)
            if not os.path.exists(newName):
                raise RuntimeError("New file should have been created '{}'.".format(newName))

        # Make sure filenames dont exceed maximum, and raises warning
        with warnings.catch_warnings(True) as ww:
            assert_equal(modify_exists(fname, max=num_files-1), None)
            assert_true(len(ww) > 0)

        # Remove created files
        for fil in rem_files:
            os.remove(fil)

        # Remove created directories
        if kill_dir:
            os.rmdir(fdir)



# Run all methods as if with `nosetests ...`
if __name__ == "__main__":
    run_module_suite()
