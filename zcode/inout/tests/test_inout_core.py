"""Test methods for `inout_core.py`.

Can be run with:
    $ nosetests zcode/inout/tests/test_inout_core.py

"""
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import warnings
import shutil
from numpy.testing import run_module_suite
import numpy as np
from nose.tools import assert_true, assert_false, assert_equal


class TestInoutCore(object):

    @classmethod
    def setup_class(cls):
        cls.fname_npz = '_test_inout_core_testfile.npz'
        cls.fname_npz_subdir = os.path.join('./subdir', cls.fname_npz)
        cls.test_dir_0 = '_test_inout_core_dir'
        cls.test_file_0 = '_test_filename.txt'
        cls._kill_test_files()

    @classmethod
    def teardown_class(cls):
        cls._kill_test_files()

    @classmethod
    def _kill_test_files(cls):
        # Remove created directories
        if os.path.exists(cls.test_dir_0):
            print("removing '{}'".format(cls.test_dir_0))
            shutil.rmtree(cls.test_dir_0)
        # Remove created files
        if os.path.exists(cls.fname_npz_subdir):
            print("removing '{}'".format(cls.fname_npz_subdir))
            os.remove(cls.fname_npz_subdir)
        if os.path.exists(cls.fname_npz):
            print("removing '{}'".format(cls.fname_npz))
            os.remove(cls.fname_npz)
        if os.path.exists(cls.test_file_0):
            print("removing '{}'".format(cls.test_file_0))
            os.remove(cls.test_file_0)

        return

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

        # Make sure subdirectories are created if needed
        dictToNPZ(data, fname_subdir)
        assert_true(os.path.exists(fname_subdir))

    def test_modify_exists_files(self):
        fdir = self.test_dir_0
        fname = self.test_file_0
        num_files = 4
        max_files = 20   # This must be between [11, 100]
        from zcode.inout.inout_core import modify_exists, modify_filename

        # Create test directory if needed, store boolean whether to later remove it.
        if not os.path.exists(fdir):
            os.makedirs(fdir)

        # Create test filename
        fname = os.path.join(fdir, fname)
        # Make sure it doesnt already exist
        if os.path.exists(fname):
            raise RuntimeError("Test filename '{}' already exists.".format(fname))

        # Create files that should *not* interfere with 'modify_exists'
        #    'modify_exists' should only look for 2-digit appended numbers
        fname_distract_1 = modify_filename(fname, append='_6')
        fname_distract_2 = modify_filename(fname, append='_123')
        print("Interference filenames = '{}', '{}'".format(fname_distract_1, fname_distract_2))
        for ff in [fname_distract_1, fname_distract_2]:
            open(ff, 'a')

        # Test that filenames are appropriately modified
        # ----------------------------------------------
        print("fname = '{}'".format(fname))
        for ii in range(num_files):
            new_name = modify_exists(fname, max=max_files)
            print(ii, "new_name = ", new_name)
            assert_false(os.path.exists(new_name))
            # Create file
            open(new_name, 'a')
            if ii == 0:
                intended_name = str(fname)
            else:
                intended_name = modify_filename(fname, append="_{:02d}".format(ii-1))

            print("\tshould be = ", intended_name)
            assert_true(os.path.exists(intended_name))
            if not os.path.exists(new_name):
                raise RuntimeError("New file should have been created '{}'.".format(new_name))

        # Make sure filenames dont exceed maximum, and raises warning
        with warnings.catch_warnings(record=True) as ww:
            assert_equal(modify_exists(fname, max=num_files-1), None)
            assert_true(len(ww) > 0)

    def test_modify_exists_dirs(self):
        fdir = self.test_dir_0
        num_files = 4
        max_files = 20   # This must be between [11, 100]
        from zcode.inout.inout_core import modify_exists, modify_filename

        # Make sure directory doesn't initially exist
        if os.path.exists(fdir) and os.path.isdir(fdir):
            shutil.rmtree(fdir)

        '''
        # Create files that should *not* interfere with 'modify_exists'
        #    'modify_exists' should only look for 2-digit appended numbers
        fname_distract_1 = modify_filename(fname, append='_6')
        fname_distract_2 = modify_filename(fname, append='_123')
        print("Interference filenames = '{}', '{}'".format(fname_distract_1, fname_distract_2))
        for ff in [fname_distract_1, fname_distract_2]:
            open(ff, 'a')
        '''

        # Test that filenames are appropriately modified
        # ----------------------------------------------
        print("fname = '{}'".format(fdir))
        for ii in range(num_files):
            new_name = modify_exists(fdir, max=max_files)
            print(ii, "new_name = ", new_name)
            assert_false(os.path.exists(new_name))
            # Create directory
            os.makedirs(new_name)
            if ii == 0:
                intended_name = str(fdir)
            else:
                intended_name = modify_filename(fdir, append="_{:02d}".format(ii-1))

            print("\tshould be = ", intended_name)
            assert_true(os.path.exists(intended_name))
            if not os.path.exists(new_name):
                raise RuntimeError("New file should have been created '{}'.".format(new_name))


# Run all methods as if with `nosetests ...`
if __name__ == "__main__":
    run_module_suite()
