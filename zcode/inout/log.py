"""Logging related classes and functions.

Classes
-------
-   IndentFormatter          - Sets the log-message indentation level based on the stack depth.

Functions
---------
-   getLogger                - Create a logger object which logs to file and or stdout stream.
-   defaultLogger            - Create a ``logging.Logger`` object which logs to the out stream.

"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import inspect
import numpy as np

__all__ = ['IndentFormatter', 'getLogger', 'defaultLogger']


class IndentFormatter(logging.Formatter):
    """Logging formatter where the depth of the stack sets the message indentation level.
    """

    def __init__(self, fmt=None, datefmt=None):
        logging.Formatter.__init__(self, fmt, datefmt)
        self.baseline = None

    def format(self, rec):
        stack = inspect.stack()
        if(self.baseline is None): self.baseline = len(stack)
        indent = (len(stack)-self.baseline)
        addSpace = ((indent > 0) & (not rec.msg.startswith(" -")))
        rec.indent = ' -'*indent + ' '*addSpace
        out = logging.Formatter.format(self, rec)
        del rec.indent
        return out


def getLogger(name, strFmt=None, fileFmt=None, dateFmt=None,
              strLevel=logging.WARNING, fileLevel=logging.DEBUG,
              tofile=None, tostr=True):
    """Create a standard logger object which logs to file and or stdout stream.

    If logging to output stream (stdout) is enabled, an `IndentFormatter` object is used.

    Arguments
    ---------
    name : str,
        Handle for this logger, must be distinct for a distinct logger.
    strFmt : str or `None`,
        Format of log messages to stream (stdout).  If `None`, default settings are used.
    fileFmt : str or `None`,
        Format of log messages to file.  If `None`, default settings are used.
    dateFmt : str or `None`
        Format of time stamps to stream and/or file.  If `None`, default settings are used.
    strLevel : int,
        Logging level for stream.
    fileLevel : int,
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

    if(tofile is None and not tostr): raise ValueError("Must log to something!")

    logger = logging.getLogger(name)
    # Make sure handlers don't get duplicated (ipython issue)
    while(len(logger.handlers) > 0): logger.handlers.pop()
    # Prevents duplication or something something...
    logger.propagate = 0

    # Determine and Set Logging Levels
    if(fileLevel is None): fileLevel = logging.DEBUG
    if(strLevel is None): strLevel = logging.WARNING
    #     Logger object must be at minimum level
    logger.setLevel(np.min([fileLevel, strLevel]))

    if(dateFmt is None): dateFmt = '%Y/%m/%d %H:%M:%S'

    # Log to file
    # -----------
    if(tofile is not None):
        if(fileFmt is None):
            fileFmt = "%(asctime)s %(levelname)8.8s [%(filename)20.20s:"
            fileFmt += "%(funcName)-20.20s]%(indent)s%(message)s"

        fileFormatter = IndentFormatter(fileFmt, datefmt=dateFmt)
        fileHandler = logging.FileHandler(tofile, 'w')
        fileHandler.setFormatter(fileFormatter)
        fileHandler.setLevel(fileLevel)
        logger.addHandler(fileHandler)
        #     Store output filename to `logger` object
        logger.filename = tofile

    # Log To stdout
    # -------------
    if(tostr):
        if(strFmt is None):
            strFmt = "%(indent)s%(message)s"

        strFormatter = IndentFormatter(strFmt, datefmt=dateFmt)
        strHandler = logging.StreamHandler()
        strHandler.setFormatter(strFormatter)
        strHandler.setLevel(strLevel)
        logger.addHandler(strHandler)

    return logger


def defaultLogger(logger=None, verbose=False, debug=False):
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

    if(isinstance(logger, logging.Logger)): return logger

    import numbers

    if(isinstance(logger, numbers.Integral)):
        level = logger
    else:
        if(debug):
            level = logging.DEBUG
        elif(verbose):
            level = logging.INFO
        else:
            level = logging.WARNING

    logger = getLogger(None, strLevel=level, tostr=True)
    return logger
