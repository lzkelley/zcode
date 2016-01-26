"""Submodule for timing components of a code.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
# from six.moves import xrange

from datetime import datetime
import numpy as np
import warnings

_DEBUG = False

__all__ = ['Timer']


class Timer(object):

    def __init__(self, errors=False):
        self.names = np.array([])
        self.durations = []
        self.starts = []
        self._errors = errors

    def __len__(self):
        if len(self.names) != (self.durations) or len(self.names) != len(self.starts):
            err_str = "lengths dont matchup, names: {}, durations: {}, starts: {}".format(
                len(self.names), len(self.durations), len(self.starts))
            if self._errors:
                raise RuntimeError(err_str)
            else:
                warnings.warn(err_str)
        return len(self.names)

    def start(self, name):
        ind = np.where(self.names == name)[0]
        # Timer already exists
        if ind.size == 1:
            # If start was already set, store duration before resetting
            if self.starts[ind] is not None:
                self.stop(name)
            # Reset new start
            self.starts[ind] = datetime.now()

        # Create new timer
        else:
            self.names = np.append(self.names, name)
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
                delta = datetime.now() - self.starts[ind]
                self.durations[ind].append(delta.total_seconds())
                self.starts[ind] = None

        # Error matching name
        else:
            err_str = "no timer for '{}'.".format(name)
            if self._errors:
                raise RuntimeError(err_str)
            else:
                warnings.warn(err_str)

    def average(self, name=None):
        if name is None:
            aves = [np.average(durs) for durs in self.durations]
        else:
            ind = np.where(self.names == name)[0]
            if ind.size == 1:
                aves = np.average(self.durations[ind])
            else:
                err_str = "No timer for '{}' found.".format(name)
                if self._errors:
                    raise RuntimeError(err_str)
                else:
                    warnings.warn(err_str)
                aves = 0.0
        return aves
