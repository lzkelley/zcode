"""Providing color-mappings from two-dimensional parameter space to RGB color-space.

Classes
-------
-   ScalarMappable2D    - Class to handle mapping a 2D parameter space to RGB color values.
Functions
---------
-   colormap2d          - Create a ScalarMappable2D to map from a 2D parameter space to RGB colors.

"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import matplotlib as mpl
from matplotlib.colors import hsv_to_rgb

import zcode.math as zmath
from . import plot_core as zplot

__all__ = ['ScalarMappable2D', 'colormap2d']


class ScalarMappable2D(object):
    """Class to handle mapping a 2D parameter space to RGB color values.

    Methods
    -------
    -   to_rgba         - Convert pairs of values to RGBA colors.
    -   legend          - Draw a legend showing the colormap on the given axes.
    """

    _LEGEND_DIV = 100
    _LEGEND_NTICKS = 4

    def __init__(self, norm, cmap, xscale='lin', yscale='lin'):
        self.norm = norm
        self.cmap = cmap
        xscale = zplot._clean_scale(xscale)
        yscale = zplot._clean_scale(yscale)
        self.xscale = xscale
        self.yscale = yscale

    def to_rgba(self, xx, yy):
        """Convert pairs of values to RGBA colors.
        """
        rgba = self.cmap(self.norm[0](xx), self.norm[1](yy))
        return rgba

    def legend(self, ax, xlabel='', ylabel=''):
        """Draw a legend showing the colormap on the given axes.
        """
        # Find extrema from normalization
        xlim = [self.norm[0].vmin, self.norm[0].vmax]
        ylim = [self.norm[1].vmin, self.norm[1].vmax]
        # Create spacings and mesh
        xx = zmath.spacing(xlim, num=self._LEGEND_DIV, scale=self.xscale)
        yy = zmath.spacing(ylim, num=self._LEGEND_DIV, scale=self.yscale)
        xx, yy = np.meshgrid(xx, yy)
        # Use the colormap to convert to RGB values
        cols = self.to_rgba(xx, yy)
        # Reshape the colors array appropriately ``(N,3)``
        cols = cols.reshape([cols.shape[0]*cols.shape[1], cols.shape[2]])
        # `pcolormesh` uses y-positions as rows, and x-pos as columns so use transpose
        im = ax.pcolormesh(xx.T, color=cols)
        # Ticks
        # -----
        # This method of using `pcolormesh` loses the x and y value data
        #     fake the limits/scaling by placing custom tick marks
        # Set x-ticks
        xticks = np.linspace(0, self._LEGEND_DIV-1, self._LEGEND_NTICKS)
        xticklabels = zmath.spacing(xlim, num=self._LEGEND_NTICKS, scale=self.xscale)
        if self.xscale.startswith('log'):
            xticklabels = [zplot.strSciNot(tt, 0, 0) for tt in xticklabels]
        else:
            xticklabels = ["{:.2f}".format(tt) for tt in xticklabels]
        # Set y-ticks
        yticks = np.linspace(0, self._LEGEND_DIV-1, self._LEGEND_NTICKS)
        yticklabels = zmath.spacing(ylim, num=self._LEGEND_NTICKS, scale=self.yscale)
        if self.yscale.startswith('log'):
            yticklabels = [zplot.strSciNot(tt, 0, 0) for tt in yticklabels]
        else:
            yticklabels = ["{:.2f}".format(tt) for tt in yticklabels]

        ax.set(xlabel=xlabel, ylabel=ylabel, xticks=xticks, yticks=yticks,
               xticklabels=xticklabels, yticklabels=yticklabels)

        return im


def _cmap2d_hsv_lin(xx, yy):
    """Linear mapping from x and y values to colors using HSV color-space.
    """
    # Define range of HSV
    HR = np.array([0.18, 1.0])
    SR = np.array([1.0, 0.5])
    VR = np.array([0.1, 1.0])

    # Convert from {0, 1} in ``xx`` and ``yy`` to HSV range
    hh = xx*np.diff(HR) + HR[0]
    vv = yy*np.diff(VR) + VR[0]
    ss = np.fabs(np.diff(SR))*np.cos(yy/(np.pi/2)) + SR[1]

    # Convert from HSV to RGB
    hsv = np.dstack((hh, ss, vv))
    rgb = hsv_to_rgb(hsv)
    rgb = rgb.reshape(np.concatenate([xx.shape, [3]]))
    return rgb


def colormap2d(xargs, yargs, cmap=None, scale=None):
    """Create a ScalarMappable2D object to map from a 2D parameter space to RGB colors.

    Arguments
    ---------
    xargs : (N,) array_like of scalars
        Values determining the extrema of the first parameter-space dimension.
    yargs : (N,) array_like of scalars
        Values determining the extrema of the second parameter-space dimension.
    cmap : NOT IMPLEMENTED YET, USE `NONE`.
    scale : one or two values, each either str or `None`

    Returns
    -------
    smap2d : `ScalarMappable2D`
        Object to handle conversion from parameter to color spaces.  Use ```to_rgba(xx, yy)``.

    """
    # Choose a default 2d mapping
    if cmap is None: cmap = _cmap2d_hsv_lin

    scale = list(np.atleast_1d(scale))
    if np.size(scale) == 1:
        scale = 2 * scale
    elif np.size(scale) != 2:
        raise ValueError("`scale` must be a single or pair of values.")

    if np.size(xargs) == 1: xargs = [0, np.int(xargs)-1]
    if np.size(yargs) == 1: yargs = [0, np.int(yargs)-1]

    if scale[0] is None: scale[0] = zmath._infer_scale(xargs)
    if scale[1] is None: scale[1] = zmath._infer_scale(yargs)

    xlog = zplot._scale_to_log_flag(scale[0])
    if xlog: xfilter = 'g'
    else:    xfilter = None

    ylog = zplot._scale_to_log_flag(scale[1])
    if ylog: yfilter = 'g'
    else:    yfilter = None

    xmin, xmax = zmath.minmax(xargs, filter=xfilter)
    ymin, ymax = zmath.minmax(yargs, filter=yfilter)

    if xlog: xnorm = mpl.colors.LogNorm(vmin=xmin, vmax=xmax)
    else:    xnorm = mpl.colors.Normalize(vmin=xmin, vmax=xmax)

    if ylog: ynorm = mpl.colors.LogNorm(vmin=ymin, vmax=ymax)
    else:    ynorm = mpl.colors.Normalize(vmin=ymin, vmax=ymax)

    smap2d = ScalarMappable2D([xnorm, ynorm], cmap, xscale=scale[0], yscale=scale[1])
    return smap2d
