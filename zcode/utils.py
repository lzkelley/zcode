"""General Utility functions for package.
"""
import warnings


def dep_warn(oldname, newname=None, msg=None, lvl=3, type='function'):
    """Standardized deprecation warning for `zcode` package.
    """
    warnings.simplefilter('always', DeprecationWarning)
    warn = "WARNING: {} `{}` is deprecated.".format(type, oldname)
    if newname is not None:
        warn += "  Use `{}` instead.".format(newname)
    if msg is not None:
        warn += "  '{}'".format(msg)

    warnings.warn(warn, DeprecationWarning, stacklevel=lvl)
    return


def dep_warn_var(old_name, old_val, new_name=None, new_val=None, msg=None, lvl=3):
    if old_val is None:
        return new_val

    warnings.simplefilter('always', DeprecationWarning)
    warn = "WARNING: {} `{}` is deprecated.".format("variable", old_name)
    if new_name is not None:
        warn += "  Use `{}` instead.".format(new_name)
    if msg is not None:
        warn += "  '{}'".format(msg)

    if new_val is not None:
        raise ValueError(warn + "  Both old and new values provided!")

    warnings.warn(warn, DeprecationWarning, stacklevel=lvl)
    return old_val
