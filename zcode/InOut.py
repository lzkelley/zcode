"""
Functions for Input/Output (IO) Operations.

Classes
-------
    StreamCapture  : class for capturing/redirecting stdout and stderr
    IndentFormatter : <logging.Formatter> subclass for stack-depth based indentation logging


Functions
---------

    statusString   :
    bytesString    :
    getFileSize    :
    countLines     :
    checkPath      : Create the given filepath if it doesn't already exist.
    dictToNPZ      :
    npzToDict      :
    getProgressBar : Wrapper to create a progressbar object with default settings.

    combineFiles   :

    getLogger      : Create a standard logger object which logs to file and/or stdout stream.
    defaultLogger  : Create a basic ``logging.Logger`` object.

    pickleSave     : Use pickle to save the target object.
    pickleLoad     : Use pickle to load from the target file.

    dillSave       : Use dill to save the target object.
    dillLoad       : Use dill to load from the target file.

    checkURL       : Check that the given url exists.

    promptYesNo    : Prompt the user (via CLI) for yes or no.

"""

import os, sys, logging, inspect, warnings
import numpy as np


class StreamCapture(list):
    """
    Class to capture/redirect output to stdout and stderr.

    See: stackoverflow.com/questions/16571150
    Usage:
       >>> with Capturing() as output:
       >>>     do_something(my_object)
       >>> print output

    """

    from cStringIO import StringIO

    def __init__(self, out=True, err=True):
        self.out = out
        self.err = err


    def __enter__(self):
        if( self.out ):
            self._stdout = sys.stdout
            sys.stdout = self._stringio = StreamCapture.StringIO()

        if( self.err ):
            self._stderr = sys.stderr
            sys.stderr = self._stringio = StreamCapture.StringIO()

        return self


    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        if( self.out ): sys.stdout = self._stdout
        if( self.err ): sys.stderr = self._stderr


# } class StreamCapture


class IndentFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None):
        logging.Formatter.__init__(self, fmt, datefmt)
        self.baseline = None
    def format(self, rec):
        stack = inspect.stack()
        if( self.baseline is None ): self.baseline = len(stack)
        indent = (len(stack)-self.baseline)
        addSpace = ((indent > 0) & (not rec.msg.startswith(" -")))
        rec.indent = ' -'*indent + ' '*addSpace
        out = logging.Formatter.format(self, rec)
        del rec.indent
        return out

# } class IndentFormatter




def statusString(count, total, durat=None):
    """
    Return a description of the status and completion of an iteration.

    If ``durat`` is provided it is used as the duration of time that the
    iterations have been going.  This time is used to estimate the time
    to completion.  ``durat`` can either be a `datetime.timedelta` object
    of if it is a scalar (int or float) then it will be converted to
    a `datetime.timedelta` object for string formatting purposes.

    Parameters
    ----------
    count : int, number of iterations completed (e.g. [0...9])
    total : int, total number of iterations (e.g. 10)
    durat : datetime.timedelta OR scalar, (optional, default=None)
    """

    # Calculate Percentage completed
    frac = 1.0*count/(total)
    stat = '%.2f%%' % (100*frac)

    if( durat is not None ):
        # Make sure `durat` is a datetime.timedelta object
        if( type(durat) is not datetime.timedelta ): durat = datetime.timedelta(seconds=durat)

        # Calculate time left
        timeLeft = 1.0*durat.total_seconds()*(1.0/frac - 1.0)
        timeLeft = np.max([timeLeft, 0.0])
        timeLeft = datetime.timedelta(seconds=timeLeft)

        # Append to status string
        stat += ' after %s, completion in ~ %s' % (str(durat), str(timeLeft))


    return stat

# statusString()


def bytesString(bytes, precision=1):
    """
    Return a humanized string representation of a number of bytes.

    Arguments
    ---------
    bytes : <scalar>, number of bytes
    precision : <int>, target precision in number of decimal places

    Returns
    -------
    strSize : <string>, human readable size

    Examples
    --------
    >> humanize_bytes(1024*12342,2)
    '12.05 MB'

    """

    abbrevs = (
        (1<<50L, 'PB'),
        (1<<40L, 'TB'),
        (1<<30L, 'GB'),
        (1<<20L, 'MB'),
        (1<<10L, 'kB'),
        (1, 'bytes')
    )

    for factor, suffix in abbrevs:
        if bytes >= factor: break

    # NOTE: for this to work right, must "from __future__ import division" else integer
    strSize = '%.*f %s' % (precision, 1.0*bytes / factor, suffix)

    return strSize

# bytesString()



def getFileSize(fnames, precision=1):
    """
    Return a human-readable size of a file or set of files.

    Arguments
    ---------
    fnames : <string> or list/array of <string>, paths to target file(s)
    precisions : <int>, desired decimal precision of output

    Returns
    -------
    byteStr : <string>, human-readable size of file(s)

    """

    ftype = type(fnames)
    if( ftype is not list and ftype is not np.ndarray ): fnames = [ fnames ]

    byteSize = 0.0
    for fil in fnames: byteSize += os.path.getsize(fil)

    byteStr = bytesString(byteSize, precision)
    return byteStr

# getFileSize()



def countLines(files, progress=False):
    """ Count the number of lines in the given file """

    # If string, or otherwise not-iterable, convert to list
    if( np.iterable(files) and not isinstance(files, str) ): files = [ files ]

    if( progress ): pbar = getProgressBar(len(files))

    nums = 0
    # Iterate over each file, count lines
    for ii,fil in enumerate(files):
        nums += sum(1 for line in open(fil))
        if( progress ): pbar.update(ii)

    if( progress ): pbar.finish()

    return nums

# countLines()


def estimateLines(files):
    """ Count the number of lines in the given file """

    if( not np.iterable(files) ): files = [files]

    lineSize = 0.0
    count = 0
    AVE_OVER = 20
    with open(files[0], 'rb') as file:
        # Average size of `AVE_OVER` lines
        for line in file:
            # Count number of bytes in line
            thisLine = len(line) // line.count(b'\n')
            lineSize += thisLine
            count += 1
            if( count >= AVE_OVER ): break

    # Convert to average line size
    lineSize /= count
    # Get total size of all files
    totalSize = sum( os.path.getsize(fil) for fil in files )
    # Estimate total number of lines
    numLines = totalSize // lineSize

    return numLines

# estimateLines()


def checkPath(tpath):
    """
    Create the given filepath if it doesn't already exist.
    """
    path,name = os.path.split(tpath)
    if( len(path) > 0 ):
        if( not os.path.isdir(path) ): os.makedirs(path)

    return path

# checkPath()



def dictToNPZ(dataDict, savefile, verbose=False):
    """
    Save the given dictionary to the given npz file.

    If the path to the given filename doesn't already exist, it is created.
    If ``verbose`` is True, the saved file size is printed out.
    """

    # Make sure path to file exists
    checkPath(savefile)

    # Save and confirm
    np.savez(savefile, **dataDict)
    if( not os.path.exists(savefile) ):
        raise RuntimeError("Could not save to file '%s'!!" % (savefile) )

    if( verbose ): print " - - Saved dictionary to '%s'" % (savefile)
    if( verbose ): print " - - - Size '%s'" % ( getFileSize(savefile) )
    return

# dictToNPZ()



def npzToDict(npz):
    """
    Given a numpy npz file or filename, convert it to a dictionary with the same keys and values.

    Arguments
    ---------
       npz <str> or <NpzFile> : input dictionary-like object

    Returns
    -------
       newDict <dict> : output dictionary with key-values from npz file.

    """
    if( type(npz) is str ): npz = np.load(npz)
    # newDict = { key : npz[key] for key in npz.keys() }
    newDict = {}
    for key in npz.keys():
        vals = npz[key]
        if( np.size(vals) == 1 and (type(vals) == np.ndarray or type(vals) == np.array) ): 
            vals = vals.item()

        newDict[key] = vals

    return newDict

# npzToDict()



def getProgressBar(maxval, width=100):
    """
    Wrapper to create a progressbar object with default settings.

    Use ``pbar.start()``, ``pbar.update(N)`` and ``pbar.finish()``
    """

    import progressbar

    # Set Progress Bar Parameters
    widgets = [
        progressbar.Percentage(),
        ' ', progressbar.Bar(),
        ' ' ]
    
    try:
        widgets.append(progressbar.AdaptiveETA())
    except:
        # warnings.warn("Could not load ``progressbar.AdaptiveETA``", RuntimeWarning)
        widgets.append(progressbar.ETA())

    # Start Progress Bar
    pbar = progressbar.ProgressBar(widgets=widgets, maxval=maxval, term_width=width)

    return pbar

# getProgressBar()



def combineFiles(inFilenames, outFilename, verbose=False):
    """
    Concatenate the contents of a set of input files into a single output file.

    Arguments
    ---------
    inFilenames : iterable<str>, list of input file names
    outFilename : <str>, output file name
    verbose : <bool> (optional=_VERBOSE), print verbose output

    Returns

    """

    # Make sure outfile path exists
    checkPath(outFilename)
    inSize = 0.0
    nums = len(inFilenames)

    # Open output file for writing
    if( verbose ): pbar = getProgressBar(nums)
    with open(outFilename, 'w') as outfil:

        # Iterate over input files
        for ii,inname in enumerate(inFilenames):
            inSize += os.path.getsize(inname)
            if( verbose ): pbar.update(ii)

            # Open input file for reading
            with open(inname, 'r') as infil:
                # Iterate over input file lines
                for line in infil: outfil.write(line)
            # } infil

        # } inname

    # } outfil

    if( verbose ): pbar.finish()

    outSize = os.path.getsize(outFilename)

    inStr   = bytesString(inSize)
    outStr  = bytesString(outSize)

    if( verbose ): print " - - - Total input size = %s, output size = %s" % (inStr, outStr)

    return

# combineFiles()


def getLogger(name, strFmt=None, fileFmt=None, dateFmt=None, strLevel=None, fileLevel=None,
              tofile=None, tostr=True):
    """
    Create a standard logger object which logs to file and or stdout stream ('str')

    Arguments
    ---------
        name    <str> : handle for this logger, must be distinct for a distinct logger

        strFmt  <str>  : format of log messages to stream (stdout)
        fileFmt <str>  : format of log messages to file
        dateFmt <str>  : format of time stamps to stream and/or file
        strLevel  <int>  : logging level for stream
        fileLevel <int>  : logging level for file
        tofile  <str>  : filename to log to (turned off if `None`)
        tostr   <bool> : log to stdout stream

    Returns
    -------
        logger  <obj>  : ``logging`` logger object

    """

    if( tofile is None and not tostr ): raise RuntimeError("Must log to something")

    logger = logging.getLogger(name)
    # Make sure handlers don't get duplicated (ipython issue)
    while len(logger.handlers) > 0: logger.handlers.pop()
    # Prevents duplication or something something...
    logger.propagate = 0

    ## Determine and Set Logging Level
    if( fileLevel is None ): fileLevel = logging.DEBUG
    if( strLevel  is None ): strLevel  = logging.WARNING
    # Logger object must be at minimum level
    logger.setLevel(np.min([fileLevel, strLevel]))

    if( dateFmt is None ): dateFmt = '%Y/%m/%d %H:%M:%S'

    ## Log to file
    #  -----------
    if( tofile is not None ):
        if( fileFmt is None ):
            fileFmt  = "%(asctime)s %(levelname)8.8s [%(filename)20.20s:"
            fileFmt += "%(funcName)-20.20s]%(indent)s%(message)s"

        fileFormatter = IndentFormatter(fileFmt, datefmt=dateFmt)
        fileHandler = logging.FileHandler(tofile, 'w')
        fileHandler.setFormatter(fileFormatter)
        fileHandler.setLevel(fileLevel)
        logger.addHandler(fileHandler)


    ## Log To stdout
    #  -------------
    if( tostr ):
        if( strFmt is None ):
            strFmt = "%(indent)s%(message)s"

        strFormatter = IndentFormatter(strFmt, datefmt=dateFmt)
        strHandler = logging.StreamHandler()
        strHandler.setFormatter(strFormatter)
        strHandler.setLevel(strLevel)
        logger.addHandler(strHandler)


    return logger

# getLogger()


def defaultLogger(logger=None, verbose=False, debug=False):
    """
    Create a basic ``logging.Logger`` object.  With no arguments, a stream-logger set to Warning.
    
    Arguments
    ---------
        logger  <obj>  : a ``logging`` level (integer), or `None` for default
        verbose <bool> : True to set 'verbose' output (`logging.INFO`)
        debug   <bool> : True to set 'debug'   output (`logging.DEBUG`), overrides ``verbose``

    Returns
    -------
        logger  <obj>  : ``logging.Logger`` object.

    """

    if( isinstance(logger, logging.Logger) ): return logger

    import numbers

    if( isinstance(logger, numbers.Integral) ):
        level = logger
    else:
        if(   debug   ): level = logging.DEBUG
        elif( verbose ): level = logging.INFO
        else:            level = logging.WARNING

    logger = getLogger(None, strLevel=level, tostr=True)

    return logger

# defaultLogger()


def pickleSave(obj, name, mode='wb'):
    """
    Use pickle to save the given object.
    
    Arguments
    ---------
        obj  <obj> : pickleable object to save
        name <str> : filename to which to save
        mode <str> : mode with which to open save file, see ``file.__doc__``

    """
    import cPickle as pickle
    
    with open(name, mode) as pfil:
        pickle.dump(obj,pfil)

    return

# pickleSave()


def pickleLoad(name, mode='rb'):
    """
    Use pickle to load the given object.
    
    Arguments
    ---------
        name <str> : filename to which to save
        mode <str> : mode with which to open save file, see ``file.__doc__``

    """
    import cPickle as pickle

    with open(name, mode) as pfil:
        pickle.load(obj,pfil)

    return

# pickleLoad()


def dillSave(obj, name, mode='wb'):
    """
    Use dill to save the given object.
    
    Arguments
    ---------
        obj  <obj> : pickleable object to save
        name <str> : filename to which to save
        mode <str> : mode with which to open save file, see ``file.__doc__``

    """
    import dill as pickle
    
    with open(name, mode) as pfil:
        pickle.dump(obj,pfil)

    return

# dillSave()


def dillLoad(name, mode='rb'):
    """
    Use dill to load the given object.
    
    Arguments
    ---------
        name <str> : filename to which to save
        mode <str> : mode with which to open save file, see ``file.__doc__``

    """
    import dill as pickle

    with open(name, mode) as pfil:
        pickle.load(obj,pfil)

    return

# dillLoad()


def checkURL(url, codemax=200, timeout=3.0):
    """
    Check that the given url exists.

    Note on ``status_code``s (see: 'https://en.wikipedia.org/wiki/List_of_HTTP_status_codes')
        1xx - informational
        2xx - success
        3xx - redirection
        4xx - client error
        5xx - server error

    """

    import requests, logging
    retval = False
    try:
        logging.getLogger("requests").setLevel(logging.WARNING)
        req = requests.head(url, timeout=timeout)
        retval = (req.status_code <= codemax)
    except:
        pass

    return retval

# } checkURL()


def promptYesNo(msg='', def='n'):
    """
    Prompt the user (via CLI) for yes or no.

    If ``def`` is 'y', then any response which *doesnt* start with 'y' will return False.
    If ``def`` is 'n', then any response which *doesnt* start with 'n' will return True.

    Arguments
    ---------
        msg <str> : message to prepend the prompt
        def <str> : default option {'y','n'}

    Returns
    -------
        retval <bool> : `True` for 'y' response, `False` for 'n'

    """

    message = str(msg)
    if( len(message) > 0 ): message += ' '

    if(   def == 'n' ): message += 'y/[n] : '
    elif( def == 'y' ): message += '[y]/n : '
    else: raise RuntimeError("Unrecognized ``def`` '%s'" % (def))

    arg = raw_input(message).strip().lower()

    if(   def == 'n' ):
        if( arg.startswith('y') ): retval = True
        else: retval = False
    elif( def == 'y' ):
        if( arg.startswith('n') ): retval = False
        else: retval = True


    return retval

# } promptYesNo()
