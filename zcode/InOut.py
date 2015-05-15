"""
Functions for Input/Output (IO) Operations.

Functions
---------

   getFileSize()
   countLines()
   checkPath()
   dictToNPZ()
   npzToDict()



"""

import os
import sys
import numpy as np



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
