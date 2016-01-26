"""Submodule for timing components of a code.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
# from six.moves import xrange

from datetime import datetime
import numpy as np
import warnings

__all__ = ['Timer', 'Timings']


class Timer(object):
    """Class for timing a single series of events.
    """

    def __init__(self, name=None):
        self.name = name
        self._start = None
        self._ave = 0.0
        self._num = 0
        self._durations = []

    def start(self, restart=False):
        """Start, or restart, the timer.
        """
        durat = None
        # If there is already a `_start` value
        if self._start is not None:
            # If we are *not* restarting (i.e. new start, without a duration)
            if not restart:
                durat = self.stop()
        # If there is no `_start` yet, or we are restarting
        self._start = datetime.now()
        return durat

    def stop(self):
        """Stop the (already started) timer.
        """
        if self._start is None:
            return None
        durat = datetime.now() - self._start
        self._durations.append(durat.total_seconds())
        self._num += 1
        # Increment cumulative average
        self._ave = self._ave + (durat - self._ave)/self._num
        return durat

    def ave(self):
        """Return the cumulative average of previously calculated durations.
        """
        return self._ave

    def durations(self):
        """Return an array of all previously calculated durations.
        """
        return np.array(self._durations)

    def last(self):
        """Return the last (most recent) calculated duration.
        """
        if self._num:
            return self._durations[-1]
        else:
            return None


class Timings(object):
    """Class for timing a set of different events, managing invidivual timers for each.
    """

    def __init__(self, errors=False):
        self._names = np.array([])
        self._timers = []

    def _ind_for_name(self, name, create=True):
        # No timer with this name exists
        if name not in self._names:
            # Create a new one
            if create:
                self.names = np.append(self.names, name)
                self._timers.append(Timer(name))
            else:
                return None

        ind = np.where(self._names == name)[0]
        # Should be a single matching name
        if ind.size != 1:
            raise RuntimeError("Name '{}' matched {} times.  Names = '{}'".format(
                name, ind.size, self.names))
        # Make sure internal name matches array name
        if self._timers[ind].name != name:
            raise RuntimeError("Names mismatch, name = '{}', timers[{}].name = '{}'".format(
                name, ind, self._timers[ind].name))

        return ind

    def start(self, name, restart=False):
        ind = self._ind_for_name(name, create=True)
        self._timers[ind].start(restart=restart)

    def stop(self, name):
        ind = self._ind_for_name(name, create=False)
        if not ind:
            raise ValueError("Timer '{}' does not exist.".format(name))
        durat = self._timers[ind].stop()
        if not durat:
            warnings.warn("Timer '{}' was not started.".format(name))
        return
