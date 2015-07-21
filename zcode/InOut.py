"""
Functions for Input/Output (IO) Operations.


Classes
-------
    StreamCapture  : class for capturing/redirecting stdout and stderr

Functions
---------

    statusString   :
    bytesString    :
    getFileSize    :
    countLines     :
    checkPath      :
    dictToNPZ      :
    npzToDict      :
    getProgressBar :

    combineFiles   :

"""

import os
import sys
import numpy as np

import warnings

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
