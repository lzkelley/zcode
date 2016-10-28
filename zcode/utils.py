"""General Utility functions for package.
"""
import warnings

__all__ = ["dep_warn"]


def dep_warn(oldname, newname=None, msg=None, lvl=3):
    """Standardized deprecation warning for `zcode` package.
    """
    warnings.simplefilter('always', DeprecationWarning)
    warn = "WARNING: `{}` is deprecated.".format(oldname)
    if newname is not None:
        warn += "  Use `{}` instead.".format(newname)
    if msg is not None:
        warn += "  '{}'".foramt(msg)

    warnings.warn(warn, DeprecationWarning, stacklevel=lvl)
    return
