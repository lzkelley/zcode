"""Functions for Input/Output (IO) Operations.

Classes
-------
-   StreamCapture            - Class to capture/redirect output to stdout and stderr.
-   Keys                     - Provide convenience for classes used as enumerated dictionary keys.

Functions
---------
-   bytesString              - Return a humanized string representation of a number of bytes.
-   getFileSize              - Return a human-readable size of a file or set of files.
-   countLines               - Count the number of lines in the given file.
-   estimateLines            - Estimate the number of lines in the given file.
-   checkPath                - Create the given filepath if it doesn't already exist.
-   dictToNPZ                - Save a dictionary to the given NPZ filename.
-   npzToDict                - Convert an NPZ file to a dictionary with the same keys and values.
-   getProgressBar           - Wrapper to create a progressbar object with default settings.
-   combineFiles             - Concatenate the contents of input files into a single output file.
-   checkURL                 - Check that the given url exists.
-   promptYesNo              - Prompt the user (via CLI) for yes or no.
-   modifyFilename           - Modify the given filename.

"""
from __future__ import absolute_import, division, print_function, unicode_literals
from six import with_metaclass

import os
import sys
import logging
import numpy as np

__all__ = ['StreamCapture', 'Keys', 'bytesString', 'getFileSize', 'countLines', 'estimateLines',
           'checkPath', 'dictToNPZ', 'npzToDict', 'getProgressBar', 'combineFiles', 'checkURL',
           'promptYesNo', 'modifyFilename']


class StreamCapture(list):
    """Class to capture/redirect output to stdout and stderr.

    See: stackoverflow.com/questions/16571150
    Usage:
       >>> with Capturing() as output:
       >>>     do_something(my_object)
       >>> print output

    """
    import io
    from io import StringIO

    def __init__(self, out=True, err=True):
        self.out = out
        self.err = err

    def __enter__(self):
        if(self.out):
            self._stdout = sys.stdout
            sys.stdout = self._stringio = StreamCapture.cStringIO.StringIO()

        if(self.err):
            self._stderr = sys.stderr
            sys.stderr = self._stringio = StreamCapture.cStringIO.StringIO()

        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        if(self.out): sys.stdout = self._stdout
        if(self.err): sys.stderr = self._stderr


class _Keys_Meta(type):
    """
    Metaclass for the ``Keys`` class.  See, ``InOut.Keys``.

    To-Do
    -----
        - Modify the attribute getter to yield more unique responses than
          the values given in the user-defined class.
          e.g.
              class TestKeys(Keys):
                  one = 1

              Then the actual value used should be something like
              ``"TestKeys.one"`` or ``"TestKeys.1"``, to make them more unique
              than just ``"one"`` or ``1``.


    """

    # Store all attribute values to list ``__values__`` (if they dont start with '__')
    def __init__(self, name, bases, dict):
        self.__init__(self)
        self.__values__ = [list(self.__dict__.values())[ii] for ii, ent in enumerate(self.__dict__)
                           if not ent.startswith('__')]

    # Iterate over the list of attributes values
    def __iter__(self):
        for val in self.__values__:
            yield val


class Keys(with_metaclass(_Keys_Meta)):
    """
    Super class to provide convenience for classes used as enumerated dictionary keys.

    Uses the metaclass ``_Key_Meta`` to override the ``__iter__`` and ``__init__`` methods.  The
    initializer simply stores a list of the *VALUES* of each attribute (not starting with '__'),
    for later use.  Iterator yields each element of the attributes values, list.

    Example
    -------
        from InOut import Keys
        class Test(Keys):
            one = '1'
            two = 'two'
            three = '3.0'

        for tt in Test:
            print tt
            if(tt == Test.two): print "Found two!"

    """


def bytesString(bytes, precision=1):
    """Return a humanized string representation of a number of bytes.

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
        (1 << 50, 'PB'),
        (1 << 40, 'TB'),
        (1 << 30, 'GB'),
        (1 << 20, 'MB'),
        (1 << 10, 'kB'),
        (1, 'bytes')
    )

    for factor, suffix in abbrevs:
        if bytes >= factor: break

    # NOTE: for this to work right, must "from __future__ import division" else integer
    strSize = '%.*f %s' % (precision, 1.0*bytes / factor, suffix)

    return strSize


def getFileSize(fnames, precision=1):
    """Return a human-readable size of a file or set of files.

    Arguments
    ---------
    fnames : <string> or list/array of <string>, paths to target file(s)
    precisions : <int>, desired decimal precision of output

    Returns
    -------
    byteStr : <string>, human-readable size of file(s)

    """

    ftype = type(fnames)
    if(ftype is not list and ftype is not np.ndarray): fnames = [fnames]

    byteSize = 0.0
    for fil in fnames: byteSize += os.path.getsize(fil)

    byteStr = bytesString(byteSize, precision)
    return byteStr


def countLines(files, progress=False):
    """Count the number of lines in the given file.
    """

    # If string, or otherwise not-iterable, convert to list
    if(np.iterable(files) and not isinstance(files, str)): files = [files]

    if(progress): pbar = getProgressBar(len(files))

    nums = 0
    # Iterate over each file, count lines
    for ii, fil in enumerate(files):
        nums += sum(1 for line in open(fil))
        if(progress): pbar.update(ii)

    if(progress): pbar.finish()

    return nums


def estimateLines(files):
    """Estimate the number of lines in the given file.
    """

    if(not np.iterable(files)): files = [files]

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
            if(count >= AVE_OVER): break

    # Convert to average line size
    lineSize /= count
    # Get total size of all files
    totalSize = sum(os.path.getsize(fil) for fil in files)
    # Estimate total number of lines
    numLines = totalSize // lineSize

    return numLines


def checkPath(tpath):
    """Create the given filepath if it doesn't already exist.
    """
    path, name = os.path.split(tpath)
    if(len(path) > 0):
        if(not os.path.isdir(path)): os.makedirs(path)

    return path


def dictToNPZ(dataDict, savefile, verbose=False):
    """Save a dictionary to the given NPZ filename.

    If the path to the given filename doesn't already exist, it is created.
    If ``verbose`` is True, the saved file size is printed out.
    """

    # Make sure path to file exists
    checkPath(savefile)

    # Save and confirm
    np.savez(savefile, **dataDict)
    if(not os.path.exists(savefile)):
        raise RuntimeError("Could not save to file '%s'!!" % (savefile))

    if(verbose): print(" - - Saved dictionary to '%s'" % (savefile))
    if(verbose): print(" - - - Size '%s'" % (getFileSize(savefile)))
    return


def npzToDict(npz):
    """Convert an NPZ file to a dictionary with the same keys and values.

    Arguments
    ---------
       npz <str> or <NpzFile> : input dictionary-like object

    Returns
    -------
       newDict <dict> : output dictionary with key-values from npz file.

    """
    if(type(npz) is str): npz = np.load(npz)
    # newDict = { key : npz[key] for key in npz.keys() }
    newDict = {}
    for key in list(npz.keys()):
        vals = npz[key]
        if(np.size(vals) == 1 and (type(vals) == np.ndarray or type(vals) == np.array)):
            vals = vals.item()

        newDict[key] = vals

    return newDict


def getProgressBar(maxval, width=100):
    """Wrapper to create a progressbar object with default settings.

    Use ``pbar.start()``, ``pbar.update(N)`` and ``pbar.finish()``
    """

    import progressbar

    # Set Progress Bar Parameters
    widgets = [
        progressbar.Percentage(),
        ' ', progressbar.Bar(),
        ' ']

    try:
        widgets.append(progressbar.AdaptiveETA())
    except:
        widgets.append(progressbar.ETA())

    # Start Progress Bar
    pbar = progressbar.ProgressBar(widgets=widgets, maxval=maxval, term_width=width)

    return pbar


def combineFiles(inFilenames, outFilename, verbose=False):
    """Concatenate the contents of a set of input files into a single output file.

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
    if(verbose): pbar = getProgressBar(nums)
    with open(outFilename, 'w') as outfil:

        # Iterate over input files
        for ii, inname in enumerate(inFilenames):
            inSize += os.path.getsize(inname)
            if(verbose): pbar.update(ii)

            # Open input file for reading
            with open(inname, 'r') as infil:
                # Iterate over input file lines
                for line in infil: outfil.write(line)

    if(verbose): pbar.finish()

    outSize = os.path.getsize(outFilename)
    inStr = bytesString(inSize)
    outStr = bytesString(outSize)

    if(verbose): print(" - - - Total input size = %s, output size = %s" % (inStr, outStr))
    return


def checkURL(url, codemax=200, timeout=3.0):
    """Check that the given url exists.

    Note on ``status_code``s (see: 'https://en.wikipedia.org/wiki/List_of_HTTP_status_codes')
        1xx - informational
        2xx - success
        3xx - redirection
        4xx - client error
        5xx - server error

    """
    import requests

    retval = False
    try:
        logging.getLogger("requests").setLevel(logging.WARNING)
        req = requests.head(url, timeout=timeout)
        retval = (req.status_code <= codemax)
    except:
        pass

    return retval


def promptYesNo(msg='', default='n'):
    """Prompt the user (via CLI) for yes or no.

    If ``default`` is 'y', then any response which *doesnt* start with 'y' will return False.
    If ``default`` is 'n', then any response which *doesnt* start with 'n' will return True.

    Arguments
    ---------
        msg <str> : message to prepend the prompt
        default <str> : default option {'y','n'}

    Returns
    -------
        retval <bool> : `True` for 'y' response, `False` for 'n'

    """

    message = str(msg)
    if(len(message) > 0): message += ' '

    if(default == 'n'): message += 'y/[n] : '
    elif(default == 'y'): message += '[y]/n : '
    else: raise RuntimeError("Unrecognized ``default`` '%s'" % (default))

    arg = input(message).strip().lower()

    if(default == 'n'):
        if(arg.startswith('y')): retval = True
        else: retval = False
    elif(default == 'y'):
        if(arg.startswith('n')): retval = False
        else: retval = True

    return retval


def modifyFilename(fname, prepend='', append=''):
    """Modify the given filename.

    Arguments
    ---------
        fname   <str> : filename to modify.
        prepend <str> : string to prepend to beginning of filename;
                        added after the terminal slash, otherwise at the beginning.
        append  <str> : string to appended to end of filename;
                        added before the terminal '.' if it exists, otherwise at the end.

    Returns
    -------
        newName <str> : new filename

    """
    oldPath, oldName = os.path.split(fname)
    newName = prepend + oldName
    if(len(append) > 0):
        oldSplit = newName.split('.')
        if(len(oldSplit) >= 2): oldSplit[-2] += append
        else:                     oldSplit[-1] += append
        newName = '.'.join(oldSplit)

    newName = os.path.join(oldPath, newName)
    return newName
