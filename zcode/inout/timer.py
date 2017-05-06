"""Submodule for timing components of a code.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

from datetime import datetime
import numpy as np
import warnings

from . import inout_core

__all__ = ['Timer', 'Timings']


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
        # Variance (standard-deviation squared)
        self._var = 0.0
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

        See: `https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance` for calculating the
             variance using an 'online algorithm'.
        """
        if self._start is None:
            return None
        durat = datetime.now() - self._start
        durat = durat.total_seconds()
        self._durations.append(durat)
        self._num += 1
        # Increment cumulative average
        prev_ave = self._ave
        self._ave = self._ave + (durat - self._ave)/self._num
        # Calculate the variance using an 'online algorithm'
        self._var = ((self._num - 1)*self._var + (durat - prev_ave)*(durat - self._ave))/self._num
        self._total += durat
        return durat

    def ave(self):
        """Return the cumulative average of previously calculated durations.
        """
        return self._ave

    def std(self):
        """Return the cumulative standard-deviation of the previously calculated durations.
        """
        # Standard-deviation is the square root of the variance
        return np.sqrt(self._var)

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
    -   start                - Start the `timer` with the given name.
    -   stop                 - Stop the timer with the given name.
    -   names                - Return an array with all of the names of the different timers.
    -   durations            - Returns an array of all the durations of the target timer.
    -   report               - Report results of durations.
    Internal:
    -   _create_timer       - Create a new timer with the given name.
    -   _ind_for_name       - The return the index corresponding to the timer of the given name.

    """

    def __init__(self, errors=False):
        # List of all individual timer names
        # self._names = np.array([])
        self._names = []
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

    def report(self, out=print):
        """Report the collected durations from all timers.

        If no internal timers exist, a warning is raised.
        If `out` is a function (e.g. `print`), then the results are outputted using that function.
        If `out` is `None`, then the results are returned as a string.

        Returns
        -------

        """
        if self._num == 0:
            warnings.warn("No timers exist.")
            return
        totals = np.array([tim.total() for tim in self._timers])
        aves = np.array([tim.ave() for tim in self._timers])
        stds = np.array([tim.std() for tim in self._timers])
        cum_tot = np.sum(totals)
        fracs = totals/cum_tot
        # Convert statistics to strings for printing
        #    Add the total fraction (1.0)
        str_fracs = np.append(fracs, np.sum(fracs))
        str_fracs = ["{:.4f}".format(fr) for fr in str_fracs]
        #    Add the total duration
        str_tots = np.append(totals, cum_tot)
        str_tots = ["{}".format(tt) for tt in str_tots]
        str_aves = ["{}".format(av) for av in aves]
        str_stds = ["{}".format(st) for st in stds]
        #    Add empty elements for overall average and standard deviation
        str_aves.append("")
        str_stds.append("")
        # Construct 2D array of results suitable for `ascii_table`
        data = np.c_[str_fracs, str_tots, str_aves, str_stds]
        cols = ['Fraction', 'Total', 'Average', 'StdDev']
        # rows = np.append(self._names, "Overall")
        rows = np.append(self._names, "Overall")
        # Print reuslts as table
        if out is None: prepend = ""
        else: prepend = "\n"
        rep = inout_core.ascii_table(data, rows=rows, cols=cols, title='Timing Results',
                                     out=out, prepend=prepend)
        return rep

    def _create_timer(self, name):
        """Create a new timer with the given name.
        """
        # self._names = np.append(self._names, name)
        self._names.append(name)
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

        # print(np.shape(self._names), np.shape(name))
        # ind = np.where(self._names == name)[0]
        # print(np.shape(ind))
        ind = self._names.index(name)
        # # Should be a single matching name
        # if ind.size != 1:
        #     raise RuntimeError("Name '{}' matched {} times.  Names = '{}'".format(
        #         name, ind.size, self._names))
        # Make sure internal name matches array name
        if self._timers[ind].name != name:
            raise RuntimeError("Names mismatch, name = '{}', timers[{}].name = '{}'".format(
                name, ind, self._timers[ind].name))

        return ind
