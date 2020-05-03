"""Notebook utility methods.
"""

import os

# import numpy as np
import astropy as ap

import sympy as sym
from sympy.physics.units.systems import SI
from sympy.physics.units import Quantity, length, mass, time  # noqa
from sympy.physics.units import c, cm, g, s, km, gravitational_constant as G  # noqa

from IPython.display import display, Math, Markdown

from zcode import plot as zplot
from zcode import math as zmath
from zcode import inout as zio


msol = Quantity("$M_\odot$")
SI.set_quantity_dimension(msol, mass)
SI.set_quantity_scale_factor(msol, ap.constants.M_sun.cgs.value*g)

pc = Quantity("pc")
SI.set_quantity_dimension(pc, length)
SI.set_quantity_scale_factor(pc, ap.constants.pc.cgs.value*cm)
kpc = Quantity("kpc")
SI.set_quantity_dimension(kpc, length)
SI.set_quantity_scale_factor(kpc, 1000*pc)
Mpc = Quantity("Mpc")
SI.set_quantity_dimension(Mpc, length)
SI.set_quantity_scale_factor(Mpc, 1e6*pc)

yr = Quantity("yr")
SI.set_quantity_dimension(yr, time)
SI.set_quantity_scale_factor(yr, ap.units.yr.cgs.scale*s)
Myr = Quantity("Myr")
SI.set_quantity_dimension(Myr, time)
SI.set_quantity_scale_factor(Myr, 1e6*yr)


def scinot(arg, acc=2, **kwargs):
    kwargs.setdefault('dollar', False)
    kwargs.setdefault('man', acc-1)
    kwargs.setdefault('exp', 1)
    kwargs.setdefault('one', False)
    return zplot.scientific_notation(arg, **kwargs)


def rm(arg):
    return "\, \\textrm{" + arg + "} \,"


def dispmath(*args):
    display(Math(*args))
    return


def dispmark(args, label=None, rm=None, mode='equation'):
    """
    dispmark(log_lum_iso, sym.latex(sym.log(lum_iso, runnoe_log_base)) + " = ")

    # mode = 'inline'
    # mode = 'equation'
    """
    tex = sym.latex(args, mode=mode, root_notation=False)
    msg = ""
    if rm is not None:
        msg += "\\textrm{{{:}}}".format(rm)
    if label is not None:
        msg += label

    tex = tex.replace("\\begin{equation}", "\\begin{{equation}}{:}".format(msg))
    display(Markdown(tex))
    return


def printm(*args, **kwargs):
    """
    printm("L_{5100} = ", exp_lum_5100.evalf(3), " = ", val_lum_5100.evalf(3))
    """
    args = [scinot(aa) if zmath.isnumeric(aa) else aa for aa in args]
    kwargs.setdefault('root_notation', False)
    tex = [sym.latex(aa, **kwargs).strip('$') for aa in args]
    tex = "$" + "".join(tex) + "$"
    display(Markdown(tex))
    return


def save_fig(fig, fname, path=None, subdir=None, modify=True, **kwargs):
    # pp = path if path is not None else PATH_OUTPUT
    pp = path if path is not None else os.path.curdir
    if subdir is not None:
        pp = os.path.join(pp, subdir, "")

    pp = zio.check_path(pp)
    ff = os.path.join(pp, fname)
    if modify:
        ff = zio.modify_exists(ff)

    ff = os.path.abspath(ff)
    kwargs.setdefault('dpi', 400)
    fig.savefig(ff, **kwargs)
    print("Saved to '{}' size: {}".format(ff, zio.get_file_size(ff)))
    return
