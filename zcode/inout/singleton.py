"""
"""
from __future__ import absolute_import, division, print_function, unicode_literals
# from six.moves import xrange

__all__ = ['Singleton']


class Singleton(object):
    """General 'singleton' object --- all created instances point to the same object.
    """
    # Create a class variable to store a single instance
    __instance = None
    # When trying to create a new instance, load old one if it exists
    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = object.__new__(cls)
        return cls.__instance
