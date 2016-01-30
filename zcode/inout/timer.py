"""Submodule for timing components of a code.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
# from six.moves import xrange

from datetime import datetime
import numpy as np
import warnings

__all__ = ['Timer', 'Timings']

from . import inout_core


class Timer(object):
    """Class for timing a single series of events.

    Methods
    -------
    -   start                - Start, or restart, this timer.
    -   stop                 - Stop this (already started) timer, store the duration.
    -   ave                  - Return the cumulative average of previously calculated durations.
    -   durations            - Return an array of all previously calculated durations.
    -   total                - Return the total, cumulative duration of all intervals.
    -   last                 - Return the duration of the last (most recent) calculated interval.

    """

    def __init__(self, name=None):
        self.name = name
        self._start = None
        self._ave = 0.0
        self._total = 0.0
        self._num = 0
        self._durations = []

    def start(self, restart=False):
        """Start, or restart, this timer.
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
        """Stop this (already started) timer, store the duration.
        """
        if self._start is None:
            return None
        durat = datetime.now() - self._start
        durat = durat.total_seconds()
        self._durations.append(durat)
        self._num += 1
        # Increment cumulative average
        self._ave = self._ave + (durat - self._ave)/self._num
        self._total += durat
        return durat

    def ave(self):
        """Return the cumulative average of previously calculated durations.
        """
        return self._ave

    def durations(self):
        """Return an array of all previously calculated durations.
        """
        return np.array(self._durations)

    def total(self):
        """Return the total, cumulative duration of all intervals.
        """
        return self._total

    def last(self):
        """Return the last (most recent) calculated duration.
        """
        if self._num:
            return self._durations[-1]
        else:
            return None


class Timings(object):
    """Class for timing a set of different events, managing invidivual timers for each.

    Methods
    -------
    -    start               - Start the `timer` with the given name.
    -    stop                - Stop the timer with the given name.
    -    names               - Return an array with all of the names of the different timers.
    -    durations           - Returns an array of all the durations of the target timer.
    Internal:
    -    _create_timer       - Create a new timer with the given name.
    -    _ind_for_name       - The return the index corresponding to the timer of the given name.


    """

    def __init__(self, errors=False):
        # List of all individual timer names
        self._names = np.array([])
        # List of all individual timers
        self._timers = []
        # The number of timers being tracked
        self._num = 0

    def start(self, name, restart=False):
        """Start the timer with the given name.

        If this timer doesnt already exist, it is created.  This is the only way to create a timer.
        """
        ind = self._ind_for_name(name, create=True)
        self._timers[ind].start(restart=restart)

    def stop(self, name):
        """Stop the timer with the given name.

        If the timer doesnt already exist, a `ValueError` is raised.
        If the timer exists, but was not yet started, a warning is raised.
        """
        ind = self._ind_for_name(name, create=False)
        if ind is None:
            raise ValueError("Timer '{}' does not exist.".format(name))
        durat = self._timers[ind].stop()
        if durat is None:
            warnings.warn("Timer '{}' was not started.".format(name))
        return

    def names(self):
        """Return an array with all of the names of the different timers.
        """
        return np.array(self._names)

    def durations(self, name):
        """Returns an array of all the durations of the target timer.

        If the timer doesnt already exist, a `ValueError` is raised.
        """
        ind = self._ind_for_name(name, create=False)
        if not ind:
            raise ValueError("Timer '{}' does not exist.".format(name))
        return self._timers[ind].durations()

    def report(self):
        """Report the collected durations from all timers.

        If no internal timers exist, a warning is raised.
        """
        if self._num == 0:
            warnings.warn("No timers exist.")
            return
        totals = np.array([tim.total() for tim in self._timers])
        cum_tot = np.sum(totals)
        fracs = totals/cum_tot
        data = np.hstack([fracs, totals])
        data = np.vstack([data, [1.0, cum_tot]])
        inout_core.ascii_table(data)
        return


    def _create_timer(self, name):
        """Create a new timer with the given name.
        """
        self._names = np.append(self._names, name)
        self._timers.append(Timer(name))
        self._num += len(self._timers)
        return

    def _ind_for_name(self, name, create=True):
        """The return the index corresponding to the timer of the given name.

        If there is no timer with the given name, and ``create == True``, then a new timer is
        created with the given name.
        """
        # No timer with this name exists
        if name not in self._names:
            # Create a new one
            if create:
                self._create_timer(name)
            else:
                return None

        ind = np.where(self._names == name)[0]
        # Should be a single matching name
        if ind.size != 1:
            raise RuntimeError("Name '{}' matched {} times.  Names = '{}'".format(
                name, ind.size, self._names))
        # Make sure internal name matches array name
        if self._timers[ind].name != name:
            raise RuntimeError("Names mismatch, name = '{}', timers[{}].name = '{}'".format(
                name, ind, self._timers[ind].name))

        return ind
