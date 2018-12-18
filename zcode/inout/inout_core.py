"""Functions for Input/Output (IO) Operations.

Classes
-------
    -   Keys                 - Provide convenience for classes used as enumerated dictionary keys.
    -   StreamCapture        - Class to capture/redirect output to stdout and stderr.

Functions
---------
    -   bytes_string         - Return a humanized string representation of a number of bytes.
    -   get_file_size        - Return a human-readable size of a file or set of files.
    -   countLines           - Count the number of lines in the given file.
    -   environment_is_jupyter - Determine if current environment is a jupyter notebook.
    -   estimateLines        - Estimate the number of lines in the given file.
    -   check_path           - Create the given filepath if it doesn't already exist.
    -   dictToNPZ            - Save a dictionary to the given NPZ filename.
    -   npzToDict            - Convert an NPZ file to a dictionary with the same keys and values.
    -   getProgressBar       - Wrapper to create a progressbar object with default settings.
    -   combineFiles         - Concatenate the contents of input files into a single output file.
    -   checkURL             - Check that the given url exists.
    -   promptYesNo          - Prompt the user (via CLI) for yes or no.
    -   modify_filename      - Modify the given filename.
    -   mpiError             - Raise an error through MPI and exit all processes.
    -   ascii_table          - Print a table with the given contents to output.
    -   modify_exists        - Modify the given filename if it already exists.
    -   python_environment   - Tries to determine the current python environment.
    -   iterable_notstring   - Return True' if the argument is an iterable and not a string type.
    -   str_format_dict      - Pretty-format a dict into a nice looking string.
    -   par_dir              - Get parent (absolute) directory name from given file/directory.
    -   top_dir              - Get the top level directory name from the given path.
    -   underline            - Add a new line of characters appended to the given string.
    -   warn_with_traceback  - Include traceback information in warnings.

"""

from __future__ import absolute_import, division, print_function, unicode_literals
import six
from datetime import datetime

import os
import sys
import re
import warnings
import numpy as np
import collections

from zcode import utils


__all__ = ['Keys', 'MPI_TAGS', 'StreamCapture', 'bytes_string', 'get_file_size',
           'countLines', 'environment_is_jupyter', 'estimateLines', 'modify_filename',
           'check_path', 'dictToNPZ', 'npzToDict', 'checkURL',
           'combine_files',
           'promptYesNo', 'mpiError', 'ascii_table', 'modify_exists', 'python_environment',
           'iterable_notstring', 'str_format_dict', 'top_dir', 'underline', 'warn_with_traceback',
           # === DEPRECATED ===
           'combineFiles']


class _Keys_Meta(type):
    """Metaclass for the ``Keys`` class.  See, ``InOut.Keys``.

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


class Keys(six.with_metaclass(_Keys_Meta)):
    """Super class to provide convenience for classes used as enumerated dictionary keys.

    Uses the metaclass ``_Key_Meta`` to override the ``__iter__`` and ``__init__`` methods.  The
    initializer simply stores a list of the *VALUES* of each attribute (not starting with '__'),
    for later use.  Iterator yields each element of the attributes values, list.

    Note:
    -   The ordering of entries is *not* preserved, and has *no* meaning.

    Example
    -------
        >>> from InOut import Keys
        >>> class Test(Keys):
        >>>     one = '1'
        >>>     two = 'two'
        >>>     three = '3.0'

        >>> for tt in Test:
        >>>     print tt
        >>>     if(tt == Test.two): print "Found two!"
        1
        3.0
        two
        Found two!

    """


class MPI_TAGS(Keys):
    """Commonly used MPI tags for master-slave paradigm.
    """
    READY = 0
    START = 1
    DONE  = 2
    EXIT  = 3


class StreamCapture(list):
    """Class to capture/redirect output to stdout and stderr.

    See: stackoverflow.com/questions/16571150
    Usage:
       >>> with Capturing() as output:
       >>>     do_something(my_object)
       >>> print output

    """
    try:
        # import for python3
        from io import StringIO
    except ImportError:
        # import for python2
        from cStringIO import StringIO

    def __init__(self, out=True, err=True):
        self.out = out
        self.err = err

    def __enter__(self):
        if(self.out):
            self._stdout = sys.stdout
            sys.stdout = self._stringio = StreamCapture.StringIO()

        if(self.err):
            self._stderr = sys.stderr
            sys.stderr = self._stringio = StreamCapture.StringIO()

        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        if(self.out): sys.stdout = self._stdout
        if(self.err): sys.stderr = self._stderr


def bytes_string(bytes, precision=1):
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
        (1 << 10, 'KB'),
        (1, 'bytes')
    )

    for factor, suffix in abbrevs:
        if bytes >= factor:
            break

    # size_str = '%.*f %s' % (precision, 1.0*bytes / factor, suffix)
    size_str = '{size:.{prec:}f} {suff}'.format(
        prec=precision, size=1.0*bytes / factor, suff=suffix)
    return size_str


def get_file_size(fnames, precision=1):
    """Return a human-readable size of a file or set of files.

    Arguments
    ---------
    fnames : str or list
        Paths to target file(s)
    precisions : int,
        Sesired decimal precision of output

    Returns
    -------
    byte_str : str
        Human-readable size of file(s)

    """
    fnames = np.atleast_1d(fnames)

    byte_size = 0.0
    for fil in fnames:
        byte_size += os.path.getsize(fil)

    byte_str = bytes_string(byte_size, precision)
    return byte_str


def countLines(files, progress=False):
    """Count the number of lines in the given file.
    """

    # If string, or otherwise not-iterable, convert to list
    if np.iterable(files) and not isinstance(files, str):
        files = [files]

    if progress:
        warnings.warn("`progress` argument is deprecated!")
        # pbar = getProgressBar(len(files))

    nums = 0
    # Iterate over each file, count lines
    for ii, fil in enumerate(files):
        nums += sum(1 for line in open(fil))
        # if(progress):
        #     pbar.update(ii)

    # if(progress):
    #     pbar.finish()

    return nums


def environment_is_jupyter():
    """Tries to determine whether the current python environment is a jupyter notebook.
    """
    return python_environment().lower().startswith('jupyter')


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


def check_path(tpath, create=True):
    """Create the given filepath if it doesn't already exist.

    Arguments
    ---------
    tpath : str
        Path to check.
    create : bool
        Create the path if it doesnt already exist.
        If `False`, and the path does *not* exist, `None` is returned

    Returns
    -------
    path : str or `None`
        If the path exists, or is created, this is the name of the directory portion of the path.
        If the path does not exist, `None` is returned

    """
    path, name = os.path.split(tpath)
    if len(path) > 0:
        if not os.path.isdir(path):
            if create:
                try:
                    os.makedirs(path)
                except FileExistsError:
                    if not os.path.isdir(path):
                        raise
            else:
                return None

    return path


def dictToNPZ(dataDict, savefile, verbose=False, log=None):
    """Save a dictionary to the given NPZ filename.

    If the path to the given filename doesn't already exist, it is created.
    If ``verbose`` is True, the saved file size is printed out.
    """
    # Make sure path to file exists
    check_path(savefile)
    # Make sure there are no scalars in the input dictionary
    for key, item in dataDict.items():
        if np.isscalar(item):
            warnStr = "Value '%s' for key '%s' is a scalar." % (str(item), str(key))
            warnings.warn(warnStr)
            dataDict[key] = np.array(item)

    # Save and confirm
    np.savez(savefile, **dataDict)
    if not os.path.exists(savefile):
        raise RuntimeError("Could not save to file '%s'." % (savefile))

    logStr = " - Saved dictionary to '{}'".format(savefile)
    logStr += " - - Size '{}'".format(get_file_size(savefile))
    try:
        log.debug(logStr)
    except Exception:
        pass

    if verbose:
        print(logStr)

    return


def npzToDict(npz):
    """Convert an NPZ file to a dictionary with the same keys and values.

    Arguments
    ---------
    npz : str or NpzFile,
        Input dictionary-like object

    Returns
    -------
    newDict : dict,
       Output dictionary with key-values from npz file.

    """

    try:
        if isinstance(npz, six.string_types):
            # Use `fix_imports` to try to resolve python2 to python3 issues.
            _npz = np.load(npz, fix_imports=True)
        else:
            _npz = npz
        newDict = _convert_npz_to_dict(_npz)

    except Exception:
        # warnings.warn("Normal load of `{}` failed ... trying different encoding".format(
        #     npz))
        if isinstance(npz, six.string_types):
            # Use `fix_imports` to try to resolve python2 to python3 issues.
            _npz = np.load(npz, fix_imports=True, encoding="bytes")
        else:
            _npz = npz
        newDict = _convert_npz_to_dict(_npz)

    return newDict


def _convert_npz_to_dict(npz):
    newDict = {}
    for key in list(npz.keys()):
        vals = npz[key]
        # Extract objects (e.g. dictionaries) packaged into size=1 arrays
        if ((np.size(vals) == 1 and (type(vals) == np.ndarray or type(vals) == np.array) and
             vals.dtype.type is np.object_)):
            vals = vals.item()

        newDict[key] = vals
    return newDict


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
    import logging

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


def modify_filename(fname, prepend='', append=''):
    """Modify the given filename.

    Arguments
    ---------
    fname : str
        Filename to modify.
    prepend : str
        String to prepend to beginning of filename.
        Added after the terminal slash, otherwise at the beginning.
    append : str
        String to appended to end of filename.
        Added before the terminal '.' if it exists, otherwise at the end.

    Returns
    -------
    new_name : str
        New filename, with modifications.

    """
    is_dir = fname.endswith('/')
    if is_dir:
        o_path, o_name = _path_fname_split(fname)
    else:
        o_path, o_name = os.path.split(fname)

    new_name = prepend + o_name
    if len(append) > 0:
        o_split = new_name.split('.')
        if (len(o_split) >= 2) and (1 < len(o_split[-1]) < 5):
            o_split[-2] += append
        else:
            o_split[-1] += append
        new_name = '.'.join(o_split)

    new_name = os.path.join(o_path, new_name)
    if is_dir:
        new_name = os.path.join(new_name, '')
    return new_name


def mpiError(comm, log=None, err="ERROR", exc_info=True):
    """Raise an error through MPI and exit all processes.

    Arguments
    ---------
       comm <...> : mpi intracommunicator object (e.g. ``MPI.COMM_WORLD``)
       err  <str> : optional, extra error-string to print

    """
    rank = comm.rank
    size = comm.size
    errStr = "\nERROR: rank %d\n%s\n%s\n" % (rank, str(datetime.now()), err)
    try:
        import traceback
        errStr += "\n" + traceback.format_exc()
    except:
        print("Couldn't use `traceback` module!")

    try:
        log.error(errStr, exc_info=exc_info)
    except:
        print(errStr)

    if(size > 1):
        comm.Abort(rank)
    else:
        raise RuntimeError(errStr)

    return


def ascii_table(table, rows=None, cols=None, title=None, out=print, linewise=False, prepend=""):
    """Print a table with the given contents to output.

    Arguments
    ---------
    table : (nx,ny) array_like of str
        2D matrix of cell values to be printed.  Must be strings.
    rows : (nx,) array_like of str or `None`
        Labels for each row.
    cols : (ny,) array_like of str or `None`
        Labels for each column.
    title : str or `None`
        Title for the whole table.  Placed in top-left corner.
    out : callable or `None`
        Method to call for output, defaults to `print`, but could also be
        for example: ``logging.Logger.debug``.
        If `None`, the table is returned as a string.
    linewise : bool
        If `True`, call the `out` comment (e.g. `print`) line-by-line.  Otherwise, call `out` on
        a single str for the entire table.
    prepend : str
        Print the given str before the table.  If ``linewise == True``, this happens for each line,
        otherwise it happens once for the entire table.

    Returns
    -------
    table : str or `None`
        If `out` is `None`, then the table is returned as a string.  Otherwise `None` is returned.

    """
    table = np.atleast_2d(table)
    nx, ny = table.shape
    if title is None: title = ''

    rows_len = 0
    if rows is not None:
        rows = np.atleast_1d(rows)
        if rows.size != nx:
            out("Length of input `rows` must match `table` shape.")
            return
        # Find longest string in `rows`
        rows_len = len(max(rows, key=len))
    # Labels must be at least as wide as title
    rows_len = len(title) if len(title) > rows_len else rows_len

    cols_len = 0
    if cols is not None:
        cols = np.atleast_1d(cols)
        if cols.size != ny:
            out("Length of input `cols` must match `table` shape.")
            return
        # Find longest string in `cols`
        cols_len = len(max(cols, key=len))

    # Cells must be at least as wide as headers (`cols`)
    cell_len = cols_len
    # Find length of longest cell
    for ii, cel in np.ndenumerate(table):
        cell_len = len(cel) if len(cel) > cell_len else cell_len

    cell_len += 1
    if rows_len > 0: rows_len += 1

    def format_cell(content, width):
        return "{1:{0}s}".format(width, content)

    # Draw Table
    # ----------
    ascii = []
    # Construct title row
    if len(title) > 0 or cols is not None:
        if cols is None: cols = ['']*ny
        row_str = format_cell(title, rows_len) + "|"
        row_str += "".join([format_cell(cc, cell_len) for cc in cols])
        row_str += "|"
        ascii.append(row_str)

    # Construct top bar
    bar_str = rows_len*"-" + "|" + ny*cell_len*"-" + "|"
    ascii.append(bar_str)

    # Construct strings for each row
    for ii, trow in enumerate(table):
        row_str = ""
        # Add row label if provided
        if rows is not None:
            row_str += "{1:{0}s}".format(rows_len, rows[ii])
        row_str += "|" + "".join(["{1:{0}s}".format(cell_len, rr) for rr in trow])
        row_str += "|"
        ascii.append(row_str)

    ascii.append(bar_str)

    table = prepend + "\n".join(ascii)
    # Return table as string
    if out is None:
        return table

    # Print table to some output
    if linewise:
        for line in ascii:
            out(prepend + line)
    else:
        out(table)

    return


def modify_exists(fname, max=1000):
    """If the given filename already exists, return a modified version.

    Returns a filename, modified by appending a 0-padded integer to the input `fname`.
    For example, if the input is 'some_dir/some_filename.txt' (assuming it already exists),
    the modified filename would be 'some_dir/some_filename_01.txt', or if that already exists,
    then 'some_dir/some_filename_02.txt' (or higher if needed, up to ``max-1``).
    Suffix numbers with the incorrect number of digits (e.g. 'some_dir/some_filename_002.txt) will
    be ignored.

    Arguments
    ---------
    fname : str
        Filename to be checked and modified.
    max : int or `None`
        Maximum number of modified filenames to try.  `None` means no limit.

    Returns
    -------
    new_name : {str, `None`}
        The input filename `fname` if it does not exist, or an appropriately modified version
        otherwise.  If the number for the new file exceeds the maximum `max`, then a warning is
        raise and `None` is returned.


    Errors
    ------
    RuntimeError is raised if:
    -   Unable to parse existing files with modified names.
    -   The new, modified filename already exists.

    Warnings
    --------
    -   The next modified filename exceeds the allowed maximum `max` number.
        In this case, `None` is returned.

    """
    # If file doesnt already exist, do nothing - return filename
    if not os.path.exists(fname):
        return fname

    is_dir = os.path.isdir(fname)

    # Determine number of digits for modified filenames to allow up to `max` files
    prec = np.int(np.ceil(np.log10(max)))

    # Look for existing, modified filenames
    # -------------------------------------
    num = 0
    # if is_dir:
    #     path, filename = _path_fname_split(fname)
    # else:
    path, filename = os.path.split(fname)
    if len(path) == 0:
        path += './'

    # construct regex for modified files
    #     look specifically for `prec`-digit numbers at the end of the filename
    # regex = modify_filename(re.escape(filename), append='_([0-9]){{{:d}}}'.format(prec))
    if is_dir:
        filename = os.path.join(filename, '')
    regex = modify_filename(re.escape(filename), append='_([0-9]){{{:d}}}'.format(prec))
    regex = regex.replace('./', '')
    if regex.endswith('/'):
        regex = regex[:-1]
    matches = sorted([ff for ff in os.listdir(path) if re.search(regex, ff)])
    # If there are matches, find the highest file-number in the matches
    if len(matches):
        mat = matches[-1]
        mat = mat.split("_")[-1]
        mat = mat.split(".")[0]
        # Try to convert to integer, raise error on failure
        try:
            num = np.int(mat)+1
        except:
            errStr = "Could not match integer from last match = '{}', mat = '{}'.".format(
                matches[-1], mat)
            raise RuntimeError(errStr)

    # If the new filename number is larger than allowed `max`, return `None`
    if num >= max:
        warnings.warn("Next number ({}) exceeds maximum ({})".format(num, max))
        return None

    # Construct new filename
    # ----------------------
    if is_dir:
        filename = os.path.join(filename, '')
    new_name = modify_filename(fname, append='_{0:0{1:d}d}'.format(num, prec))

    # New filename shouldnt exist; if it does, raise warning
    if os.path.exists(new_name):
        # raise RuntimeError("New filename '{}' already exists.".format(new_name))
        warnings.warn("New filename '{}' already exists.".format(new_name))
        return modify_exists(new_name)

    return new_name


def python_environment():
    """Tries to determine the current python environment, one of: 'jupyter', 'ipython', 'terminal'.
    """
    try:
        # NOTE: `get_ipython` should not be explicitly imported from anything
        ipy_str = str(type(get_ipython())).lower()  # noqa
        # print("ipy_str = '{}'".format(ipy_str))
        if 'zmqshell' in ipy_str:
            return 'jupyter'
        if 'terminal' in ipy_str:
            return 'ipython'
    except:
        return 'terminal'


def iterable_notstring(var):
    """Return True' if the argument is an iterable and not a string type.
    """
    return not isinstance(var, six.string_types) and isinstance(var, collections.Iterable)


def str_format_dict(jdict, **kwargs):
    """Pretty-format a dictionary into a nice looking string using the `json` package.

    Arguments
    ---------
    jdict : dict,
        Input dictionary to be formatted.

    Returns
    -------
    jstr : str,
        Nicely formatted string.

    """
    kwargs.setdefault('sort_keys', True)
    kwargs.setdefault('indent', 4)
    import json
    jstr = json.dumps(jdict, separators=(',', ': '), **kwargs)
    return jstr


def top_dir(idir):
    """Get the top level directory name from the given path.

    e.g. top_dir("/Users/lzkelley/Programs/zcode/zcode/")             -> "zcode"
         top_dir("/Users/lzkelley/Programs/zcode/zcode")              -> "zcode"
         top_dir("/Users/lzkelley/Programs/zcode/zcode/constants.py") -> "zcode"
    """
    # Removing trailing slash if included
    if idir.endswith('/'):
        idir = idir[:-1]

    # If this is already a directory, then `basename` is the top-level directory
    if os.path.isdir(idir):
        top = os.path.basename(idir)
    # Otherwise, split path first
    else:
        idir, fil = os.path.split(idir)
        top = os.path.basename(idir)

    return top


def underline(in_str, char=None):
    """Return a copy of the input string with a new line of '-' appended, with matching length.
    """
    if char is None:
        char = '-'
    use_str = in_str.split("\n")[-1]
    out_str = in_str + "\n" + char*len(use_str)
    return out_str


def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    """Use this method in place of `warnings.showwarning` to include traceback information.

    Use:
        `warnings.showwarning = warn_with_traceback`

    Taken from: http://stackoverflow.com/a/22376126/230468
    """
    import traceback
    traceback.print_stack()
    log = file if hasattr(file, 'write') else sys.stderr
    log.write(warnings.formatwarning(message, category, filename, lineno, line))
    return


def _path_fname_split(fname):
    """
    """
    path, filename = os.path.split(fname)
    # Make sure `filename` stores directory names if needed
    #    If a `fname` looks like "./dname/", then split yields ('./dname', '')
    #    convert this to ('', './dname')
    # print("\t", path, filename)
    if len(filename) == 0 and len(path) > 0:
        filename = path
        path = ''
    # convert ('', './dname') --> ('./', 'dname')
    if filename.startswith('./'):
        path = filename[:2]
        filename = filename[2:]

    # Either path should have a path stored, or it should be the local directory
    if len(path) == 0:
        path = './'
    # print("\t", path, filename)
    return path, filename


def combine_files(inFilenames, outFilename, verbose=False):
    """Concatenate the contents of a set of input files into a single output file.

    Arguments
    ---------
    inFilenames : iterable<str>, list of input file names
    outFilename : <str>, output file name
    verbose : <bool> (optional=_VERBOSE), print verbose output

    Returns

    """

    # Make sure outfile path exists
    check_path(outFilename)
    inSize = 0.0
    # nums = len(inFilenames)

    # Open output file for writing
    if verbose:
        warnings.warn("`progress` is deprecated!")
        # pbar = getProgressBar(nums)
    with open(outFilename, 'w') as outfil:

        # Iterate over input files
        for ii, inname in enumerate(inFilenames):
            inSize += os.path.getsize(inname)
            # if verbose:
            #     pbar.update(ii)

            # Open input file for reading
            with open(inname, 'r') as infil:
                # Iterate over input file lines
                for line in infil: outfil.write(line)

    # if verbose:
    #     pbar.finish()

    outSize = os.path.getsize(outFilename)
    inStr = bytes_string(inSize)
    outStr = bytes_string(outSize)

    if verbose:
        print("Total input size = %s, output size = %s" % (inStr, outStr))
    return


def combineFiles(*args, **kwargs):
    utils.dep_warn("combineFiles", newname="combine_files")
    return combine_files(*args, **kwargs)
