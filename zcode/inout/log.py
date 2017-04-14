"""Logging related classes and functions.

Classes
-------
-   IndentFormatter          - Sets the log-message indentation level based on the stack depth.

Functions
---------
-   get_logger                - Create a logger object which logs to file and or stdout stream.
-   default_logger            - Create a ``logging.Logger`` object which logs to the out stream.

"""
from __future__ import absolute_import, division, print_function, unicode_literals

from datetime import datetime
import logging
import shutil
import inspect
import numpy as np

from . import inout_core
from .. import utils

__all__ = ['IndentFormatter', 'get_logger', 'default_logger',
           # DEPRECATED:
           'getLogger', 'defaultLogger']


class IndentFormatter(logging.Formatter):
    """Logging formatter where the depth of the stack sets the message indentation level.
    """

    def __init__(self, fmt=None, format_date=None):
        logging.Formatter.__init__(self, fmt, format_date)
        self.baseline = None

    def format(self, rec):
        stack = inspect.stack()
        if self.baseline is None: self.baseline = len(stack)
        indent = (len(stack)-self.baseline)
        addSpace = ((indent > 0) & (not rec.msg.startswith(" -")))
        rec.indent = ' -'*indent + ' '*addSpace
        out = logging.Formatter.format(self, rec)
        del rec.indent
        return out


def getLogger(*args, **kwargs):
    utils.dep_warn("getLogger", newname="get_logger")
    return get_logger(*args, **kwargs)


def get_logger(name, format_stream=None, format_file=None, format_date=None,
               level_stream=logging.WARNING, level_file=logging.DEBUG,
               tofile=None, tostr=True):
    """Create a standard logger object which logs to file and or stdout stream.

    If logging to output stream (stdout) is enabled, an `IndentFormatter` object is used.

    Arguments
    ---------
    name : str,
        Handle for this logger, must be distinct for a distinct logger.
    format_stream : str or `None`,
        Format of log messages to stream (stdout).  If `None`, default settings are used.
    format_file : str or `None`,
        Format of log messages to file.  If `None`, default settings are used.
    format_date : str or `None`
        Format of time stamps to stream and/or file.  If `None`, default settings are used.
    level_stream : int,
        Logging level for stream.
    level_file : int,
        Logging level for file.
    tofile : str or `None`,
        Filename to log to (turned off if `None`).
    tostr : bool,
        Log to stdout stream.

    Returns
    -------
    logger : ``logging.Logger`` object,
        Logger object to use for logging.

    """

    if (tofile is None) and (not tostr):
        raise ValueError("Must log to something!")

    logger = logging.getLogger(name)
    # Make sure handlers don't get duplicated (ipython issue)
    while len(logger.handlers) > 0:
        logger.handlers.pop()
    # Prevents duplication or something something...
    logger.propagate = 0

    # Determine and Set Logging Levels
    if level_file is None:
        level_file = logging.DEBUG
    if level_stream is None:
        level_stream = logging.WARNING
    #     Logger object must be at minimum level (`np` int doesnt work, need regular int)
    logger.setLevel(int(np.min([level_file, level_stream]).astype(int)))

    if format_date is None:
        format_date = '%Y/%m/%d %H:%M:%S'

    # Log to file
    # -----------
    if tofile is not None:
        if format_file is None:
            format_file = "%(asctime)s %(levelname)8.8s [%(filename)20.20s:"
            format_file += "%(funcName)-20.20s]%(indent)s%(message)s"

        fileFormatter = IndentFormatter(format_file, format_date=format_date)
        fileHandler = logging.FileHandler(tofile, 'w')
        fileHandler.setFormatter(fileFormatter)
        fileHandler.setLevel(level_file)
        logger.addHandler(fileHandler)
        #     Store output filename to `logger` object
        logger.filename = tofile

    # Log To stdout
    # -------------
    if tostr:
        if format_stream is None:
            format_stream = "%(indent)s%(message)s"

        strFormatter = IndentFormatter(format_stream, format_date=format_date)
        strHandler = logging.StreamHandler()
        strHandler.setFormatter(strFormatter)
        strHandler.setLevel(level_stream)
        logger.addHandler(strHandler)

    # Add a `raise` method to both log an error and raise one
    # -------------------------------------------------------
    def _raise(self, msg, error=RuntimeError):
        self.error(msg)
        raise error(msg)
    # Not entirely sure why this works, but it seems to
    logger.raise_error = _raise.__get__(logger)

    # Add a `after` method to log how long something took
    # ---------------------------------------------------
    logger._after_lvl = logging.INFO

    def _after(self, msg, beg, beg_all=None, lvl=None):
        if lvl is None:
            lvl = self._after_lvl
        _str = "{} after {}".format(msg, datetime.now()-beg)
        if beg_all is not None:
            _str += " ({})".format(datetime.now()-beg_all)
        self.log(lvl, _str)
    # Not entirely sure why this works, but it seems to
    logger.after = _after.__get__(logger)

    # Add a `copy_file` method to copy logfile to the given destination
    # -----------------------------------------------------------------
    def _copy(self, dest, modify_exists=False):
        if modify_exists:
            dest = inout_core.modify_exists(dest)
        inout_core.check_path(dest)
        shutil.copy(self.filename, dest)
    # Not entirely sure why this works, but it seems to
    logger.copy = _copy.__get__(logger)

    return logger


def defaultLogger(*args, **kwargs):
    utils.dep_warn("defaultLogger", newname="default_logger")
    return default_logger(*args, **kwargs)


def default_logger(logger=None, verbose=False, debug=False):
    """Create a basic ``logging.Logger`` object which logs to the out stream.

    Arguments
    ---------
    logger : ``logging.Logger`` object, int or `None`,
        This can be an existing logger object (in which case nothing happens),
        a ``logging`` level (integer), or `None` for default settings.
    verbose : bool,
        True to set 'verbose' output (``logging.INFO``)
    debug : bool
        True to set 'debug' output (``logging.DEBUG``), overrides `verbose`.

    Returns
    -------
    logger : ``logging.Logger`` object,
        Resulting logger object.

    """

    if isinstance(logger, logging.Logger):
        return logger

    import numbers

    if isinstance(logger, numbers.Integral):
        level = logger
    else:
        if debug:
            level = logging.DEBUG
        elif verbose:
            level = logging.INFO
        else:
            level = logging.WARNING

    logger = get_logger(None, level_stream=level, tostr=True)
    return logger
