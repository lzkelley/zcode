"""Functions for dealing with units of time (and datetimes).

Functions
---------
-   datetime_to_decimal_year           -
-   to_datetime                        -

"""
import datetime
import astropy as ap
import astropy.time
import dateutil
import dateutil.parser
import warnings


__all__ = ['to_decimal_year', 'to_datetime', 'to_str']


def to_datetime(dt, format=None):
    """Convert the given object into a `datetime.datetime` instance.

    Arguments
    ---------
    dt : str, float, or `datetime.datetime` or `astropy.time.Time`,
        Datetime specification.
        - `str`:
            - If a str is given with a `format` specification, then `datetime.datetime.strptime` is
              used to parse `dt`.
            - If a str is given without `format`, then first `ap.time.Time` is called on the given
              str, and if that fails, then `dateutil.parser.parse` is called.
        - `float`: this is assumed to be a 'unix' time specifications (seconds after 1970/01/01).
    format : str or `None`
        If `format` is not `None`, then it is interpretted as the specifications for parsing the
        datetime using the `datetime.datetime.strptime` method.

    Returns
    -------
    dt : `datetime.datetime`
        Datetime instance.
    """
    if isinstance(dt, datetime.datetime):
        return dt

    if isinstance(dt, ap.time.Time):
        return dt.datetime

    if format is not None:
        dt = datetime.datetime.strptime(dt, format)
        return dt

    if isinstance(dt, float):
        # astropy throws errors for times in the far future; catch them
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            dt = ap.time.Time(dt, format='unix').to_datetime()
        return dt

    try:
        # astropy throws errors for times in the far future; catch them
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            dt = ap.time.Time(dt).to_datetime()
    except ValueError:
        dt = dateutil.parser.parse(dt)

    return dt


def to_decimal_year(dt, **kwargs):
    """Convert the given datetime into a decimal year (down to millisecond precision).
    """
    # Convert string to datetime if needed
    # if not isinstance(dt, datetime.datetime):
    #     dt = datetime.datetime.strptime(dt, format)
    dt = to_datetime(dt, **kwargs)
    temp = dt.strftime("%Y|%j|%H|%M|%S.%f")
    year, day, hr, mnt, sec = [float(tt) for tt in temp.split('|')]
    hours = hr + mnt/60.0 + sec/3600.0
    max_day = int(datetime.datetime(int(year), 12, 31, 23, 59, 59).strftime("%j"))
    # print("year: {}, day: {}, max_day: {}".format(year, day, max_day))
    dec_yr = 1.0*year + (day - 1.0 + (hours/24.0)) / max_day
    return dec_yr


def to_str(dt, format="%Y-%m-%d %H:%M:%S.%f"):
    """Convert the given datetime specification into a formatted string.

    Arguments
    ---------
    dt : object or `None`,
        A datetime specification.  See `zcode.math.time.to_datetime()`.
        If `None`, then the 'None' str is returned.
    format : str,
        Specification for how to format the datetime.  Must be a `strftime` style specification.

    Returns
    -------
    str_time : str
        Str representation of the give datetime.

    """
    if dt is None:
        return str(dt)
    dt = to_datetime(dt)
    # Convert to str
    str_time = dt.strftime(format)
    return str_time
