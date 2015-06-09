"""
Functions for Input/Output (IO) Operations.

Functions
---------

   statusString() :
   bytesString()  :
   getFileSize()  :
   countLines()   :
   checkPath()    :
   dictToNPZ()    :
   npzToDict()    :



"""

import os
import sys
import numpy as np



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
    if( not iterableNotString(files) ): files = [ files ]

    if( progress ):
        numFiles = len(files)
        if( numFiles < 100 ): interval = 1
        else:                 interval = np.int(np.floor(numFiles/100.0))
        start = datetime.datetime.now()

    nums = 0
    # Iterate over each file
    for ii,fil in enumerate(files):
        # Count number of lines
        nums += sum(1 for line in open(fil))

        # Print progresss
        if( progress ):
            now = datetime.datetime.now()
            dur = now-start

            statStr = aux.statusString(ii+1, numFiles, dur)
            sys.stdout.write('\r - - - %s' % (statStr))
            sys.stdout.flush()
            if( ii+1 == numFiles ): sys.stdout.write('\n')


    return nums


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
    Given a numpy npz file, convert it to a dictionary with the same keys and values.

    Arguments
    ---------
    npz : <NpzFile>, input dictionary-like object

    Returns
    -------
    newDict : <dict>, output dictionary with key-values from npz file.

    """
    if( type(npz) is str ): npz = np.load(npz)
    newDict = { key : npz[key] for key in npz.keys() }
    return newDict

# npzToDict()
