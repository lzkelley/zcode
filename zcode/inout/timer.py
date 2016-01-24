"""Submodule for timing components of a code.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from six.moves import xrange

from datetime import datetime
import numpy as np
import warnings

_DEBUG = False

__all__ = ['Timer']


class Timer(object):

    def __init__(self, errors=False):
        self.names = np.array([])
        self.durations = np.array([])
        self.starts = np.array([])
        self._errors = errors

    def start(self, name):
        ind = np.where(self.names == name)[0]
        # Timer already exists
        if ind.size == 1:
            # If start was already set, store duration before resetting
            if self.starts[ind] is not None:
                self.durations[ind].append(datetime.now() - self.starts[ind])
            # Reset start
            self.starts[ind] = datetime.now()

        # Create new timer
        else:
            self.names.append(name)
            self.starts.append(datetime.now())
            self.durations.append([])
            if _DEBUG:
                print("<Timer>: lengths = {}, {}, {}".format(
                    len(self.names), len(self.starts), len(self.durations)))

    def stop(self, name):
        ind = np.where(self.names == name)[0]
        # If there is a matching timer
        if ind.size == 1:
            # If this timer wasnt started, store zero and raise warning.
            if self.starts[ind] is None:
                err_str = "timer for '{}' was not started, storing zero duration.".format(name)
                if self._errors:
                    raise RuntimeError(err_str)
                else:
                    warnings.warn(err_str)
                self.durations[ind].append(0.0)
            # If it was started, store duration, reset start
            else:
                self.durations[ind].append(datetime.now() - self.starts[ind])
                self.starts[ind] = None

        # Error matching name
        else:
            err_str = "no timer for '{}'.".format(name)
            if self._errors:
                raise RuntimeError(err_str)
            else:
                warnings.warn(err_str)
