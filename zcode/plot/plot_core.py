"""General plotting functions.

Functions
---------
-   setAxis              - Set many different axis properties at once.
-   twinAxis             - Easily create and set a new twin axis (like `twinx()` or `twiny()`)
-   set_lim              - Set limits on an axis
-   set_ticks
-   zoom                 - Zoom-in at a certain location on the given axes.
-   stretchAxes          - Stretch the `x` and/or `y` limits of the given axes by a scaling factor.
-   text                 - Add text to figure.
-   label_line           - Add text to line
-   legend               - Add a legend to the given figure.
-   unifyAxesLimits      - Set limits on all given axes to match global extrema.
-   color_cycle          - Create a range of colors.
-   colormap             - Create a colormap from scalars to colors.
-   cut_colormap         - Select a truncated subset of the given colormap.
-   color_set            - Retrieve a (small) set of color-strings with hand picked values.
-   set_grid             - Configure the axes' grid.
-   skipTicks            - skip some tick marks
-   saveFigure           - Save the given figure(s) to the given filename.
-   strSciNot            - Convert a scalar into a string with scientific notation.
-   line_style_set       - Retrieve a set of line-style specifications.

-   plotHistLine         - Plot a histogram line.
-   plotSegmentedLine    - Draw a line segment by segment.
-   plotScatter          - Draw a scatter plot.
-   plotHistBars         - Plot a histogram bar graph.
-   plotConfFill         - Draw a median line and (set of) confidence interval(s).
-   line_label           - Plot a vertical line, and give it a label outside the axes.
-   full_extent          -
-   position_to_extent   -
-   backdrop             -

_   _extents             -
-   _set_extents         -
-   _setAxis_scale       -
-   _setAxis_label       -
-   _histLine            - construct a stepped line
-   _clear_frame         -
-   _make_segments       -
-   _scale_to_log_flag   -
-   _clean_scale         -
-   _get_cmap            - Retrieve a colormap with the given name if it is not already a colormap.

"""
from __future__ import absolute_import, division, print_function, unicode_literals
import six
from six.moves import xrange

import os
import logging
import numbers
import warnings

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn.apionly as sns

import zcode.math as zmath
import zcode.inout as zio
from zcode import utils

__all__ = ['setAxis', 'twinAxis', 'set_lim', 'set_ticks', 'zoom',
           'stretchAxes', 'text', 'label_line', 'legend',
           'unifyAxesLimits', 'color_cycle', 'transform',
           'colorCycle', 'colormap', 'color_set', 'set_grid',
           'skipTicks', 'saveFigure', 'strSciNot',
           'plotHistLine', 'plotSegmentedLine', 'plotScatter',
           'plotHistBars', 'plotConfFill', 'line_style_set',
           'line_label', 'full_extent', 'position_to_extent',
           'backdrop', '_histLine', '_scale_to_log_flag',
           # Deprecated
           'setGrid', 'setLim'
           ]

COL_CORR = 'royalblue'
LW_CONF = 1.0
VALID_SIDES = [None, 'left', 'right', 'top', 'bottom']
_COLOR_SET = ['blue', 'red', 'green', 'purple',
              'orange', 'cyan', 'brown', 'gold', 'pink',
              'forestgreen', 'grey', 'olive', 'coral', 'yellow']

_COLOR_SET_XKCD = ["blue", "red", "green", "purple", "cyan", "orange",
                   "pink", "brown", "magenta", "amber", "slate blue",
                   "teal", "light blue", "lavender", "rose", "turquoise", "azure",
                   "lime green", "greyish", "windows blue",
                   "faded green", "mustard", "brick red", "dusty purple"]

_LINE_STYLE_SET = [
    [],
    [8, 4],
    [1, 1],
    [8, 1, 1, 1],
    [8, 1, 1, 1, 1, 1],
    [8, 1, 1, 1, 1, 1, 1, 1],
    [8, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [4, 2],
    [8, 2, 4, 2],
    [8, 2, 4, 2, 1, 2],
    [4, 1, 1, 1],
    [4, 1, 1, 1, 1, 1],
    [4, 1, 1, 1, 1, 1, 1, 1],
    [4, 1, 1, 1, 1, 1, 1, 1, 1, 1],
]

_LW_OUTLINE = 0.8
_PAD = 0.01

# Default length for lines in legend handles; in units of font-size
_HANDLE_LENGTH = 2.5
_SCATTER_POINTS = 1


def setAxis(ax, axis='x', c='black', fs=12, pos=None, trans='axes', label=None, scale=None,
            thresh=None, side=None, ts=8, grid=True, lim=None, invert=False, ticks=True,
            stretch=1.0):
    """
    Configure a particular axis of the given axes object.

    Arguments
    ---------
       ax     : <matplotlib.axes.Axes>, base axes object to modify
       axis   : <str>, which axis to target {``x`` or ``y``}
       c      : <str>, color for the axis (see ``matplotlib.colors``)
       fs     : <int>, font size for labels
       pos    : <float>, position of axis-label/lines relative to the axes object
       trans  : <str>, transformation type for the axes
       label  : <str>, axes label (``None`` means blank)
       scale  : <str>, axis scale, e.g. 'log', (``None`` means default)
       thresh : <float>, for 'symlog' scaling, the threshold for the linear segment
       side   : <str>, where to place the markings, {``left``, ``right``, ``top``, ``bottom``}
       ts     : <int>, tick-size (for the major ticks only)
       grid   : <bool>, whether grid lines should be enabled
       lim    : <float>[2], limits for the axis range
       invert : <bool>, whether to invert this axis direction (i.e. high to low)
       ticks
       stretch : <flt>,

    """

    assert axis in ['x', 'y'], "``axis`` must be `x` or `y`!"
    assert trans in ['axes', 'figure'], "``trans`` must be `axes` or `figure`!"
    assert side in VALID_SIDES, "``side`` must be in '%s'" % (VALID_SIDES)

    # Set tick colors and font-sizes
    ax.tick_params(axis=axis, which='both', colors=c, labelsize=fs)
    #    Set tick-size only for major ticks
    ax.tick_params(axis=axis, which='major', size=ts)

    # Set Grid Lines
    set_grid(ax, grid, axis='both')

    if axis == 'x':
        ax.xaxis.label.set_color(c)
        offt = ax.get_xaxis().get_offset_text()

        if side is None:
            if pos is None:
                side = 'bottom'
            else:
                if pos < 0.5:
                    side = 'bottom'
                else:
                    side = 'top'

        if pos is not None:
            offt.set_y(pos)
            ax.xaxis.set_label_position(side)
            ax.xaxis.set_ticks_position(side)

        if lim is not None:
            if np.size(lim) > 2: lim = zmath.minmax(lim)
            ax.set_xlim(lim)

        if invert: ax.invert_xaxis()
        if not ticks:
            for tlab in ax.xaxis.get_ticklabels(): tlab.set_visible(False)

    else:
        ax.yaxis.label.set_color(c)
        offt = ax.get_yaxis().get_offset_text()

        if side is None:
            if pos is None:
                side = 'left'
            else:
                if pos < 0.5:
                    side = 'left'
                else:
                    side = 'right'

        if pos is not None:
            offt.set_x(pos)

        ax.yaxis.set_label_position(side)
        ax.yaxis.set_ticks_position(side)

        if lim is not None: ax.set_ylim(lim)

        if invert:
            ax.invert_yaxis()
        if not ticks:
            for tlab in ax.yaxis.get_ticklabels(): tlab.set_visible(False)

    # Set Spine colors
    ax.spines[side].set_color(c)
    if pos is not None:
        ax.set_frame_on(True)
        ax.spines[side].set_position((trans, pos))
        ax.spines[side].set_visible(True)
        ax.patch.set_visible(False)

    # Set Axis Scaling
    if scale is not None: _setAxis_scale(ax, axis, scale, thresh=thresh)

    # Set Axis Label
    _setAxis_label(ax, axis, label, fs=fs, c=c)

    if stretch != 1.0:
        if axis == 'x': ax = stretchAxes(ax, xs=stretch)
        elif axis == 'y': ax = stretchAxes(ax, ys=stretch)

    offt.set_color(c)
    return ax


def twinAxis(ax, axis='x', pos=1.0, **kwargs):
    """
    """
    if axis == 'x':
        tw = ax.twinx()
        setax = 'y'
    elif axis == 'y':
        tw = ax.twiny()
        setax = 'x'
    else:
        raise RuntimeError("``axis`` must be either {`x` or `y`}!")

    tw = setAxis(tw, axis=setax, pos=pos, **kwargs)
    return tw


def setLim(*args, **kwargs):
    utils.dep_warn("setLim", newname="set_lim")
    return set_lim(*args, **kwargs)


def set_lim(ax, axis='y', lo=None, hi=None, data=None, range=False, at='exactly', invert=False):
    """Set the limits (range) of the given, target axis.

    When only ``lo`` or only ``hi`` is specified, the default behavior is to only set that axis
    limit and leave the other bound to its existing value.  When ``range`` is set to `True`, then
    the given axis boumds (``lo``/``hi``) are used as multipliers, i.e.

        >>> Plotting.set_lim(ax, lo=0.1, range=True, at='exactly')
        will set the lower bound to be `0.1` times the existing upper bound

    The ``at`` keyword determines whether the given bounds are treated as limits to the bounds,
    or as fixed ('exact') values, i.e.

        >>> Plotting.set_lim(ax, lo=0.1, range=True, at='most')
        will set the lower bound to at-'most' `0.1` times the existing upper bound.  If the lower
        bound is already 0.05 times the upper bound, it will not be changed.


    Arguments
    ---------
       ax    : <matplotlib.axes.Axes>, base axes object to modify
       axis  : <str>{'x','y'}, which axis to set
       lo    : <scalar>, lower  limit bound
       hi    : <scalar>, higher (upper) limit bound
       data  : <scalar>[N], range of data values from which to use max and min
       range : <bool>, set the 'range' of the axis limits (True) or set the bounds explicitly
       at    : <str>{'least', 'exactly', 'most'}, how to treat the given bounds - limits or exactly

    """

    AT_LEAST = 'least'
    AT_MOST = 'most'
    AT_EXACTLY = 'exactly'
    AT_VALID = [AT_LEAST, AT_EXACTLY, AT_MOST]
    assert at in AT_VALID, "``at`` must be in {'%s'}!" % (str(AT_VALID))

    if axis == 'y':
        get_lim = ax.get_ylim
        set_lim = ax.set_ylim
    elif axis == 'x':
        get_lim = ax.get_xlim
        set_lim = ax.set_xlim
    else:
        raise RuntimeError("``axis`` must be either 'x' or 'y'!")

    lims = np.array(get_lim())

    # Set Range/Span of Limits
    if range:
        if lo is not None:
            if at == AT_EXACTLY:
                lims[0] = lims[1]/lo
            elif at == AT_LEAST:
                lims[0] = np.max([lims[0], lims[0]/lo])
            elif at == AT_MOST:
                lims[0] = np.min([lims[0], lims[0]/lo])
        elif hi is not None:
            if at == AT_EXACTLY:
                lims[1] = lims[1]*hi
            elif at == AT_LEAST:
                lims[1] = np.max([lims[1], lims[1]*hi])
            elif at == AT_MOST:
                lims[1] = np.min([lims[1], lims[1]*hi])
        else:
            raise RuntimeError("``lo`` or ``hi`` must be provided!")

    # Set Limits explicitly
    else:
        if lo is not None:
            if at == AT_EXACTLY:
                lims[0] = lo
            elif at == AT_LEAST:
                lims[0] = np.max([lims[0], lo])
            elif at == AT_MOST:
                lims[0] = np.min([lims[0], lo])
            else:
                raise ValueError("Unrecognized `at` = '%s'" % (at))
        elif data is not None:
            lims[0] = np.min(data)

        if hi is not None:
            if at == AT_EXACTLY:
                lims[1] = hi
            elif at == AT_LEAST:
                lims[1] = np.max([lims[1], hi])
            elif at == AT_MOST:
                lims[1] = np.min([lims[1], hi])
            else:
                raise ValueError("Unrecognized `at` = '%s'" % (at))
        elif data is not None:
            lims[1] = np.max(data)

    # Actually set the axes limits
    set_lim(lims)
    if invert:
        if axis == 'x': ax.invert_xaxis()
        else:            ax.invert_yaxis()

    return


def set_ticks(ax, axis='y', every=2, log=True):
    """DEV
    """
    if axis != 'y': raise ValueError("Only 'y' axis currently supported.")
    if not log: raise ValueError("Only `log` scaling currently supported.")

    ylims = np.array(ax.get_ylim())
    man, exp = zmath.frexp10(ylims[0])
    low = np.int(exp)
    man, exp = zmath.frexp10(ylims[1])
    high = np.int(exp)

    vals = np.arange(low, high, every)
    vals = np.power(10.0, vals)
    ax.set_yticks(vals)
    return


def zoom(ax, loc, axis='x', scale=2.0):
    """Zoom-in at a certain location on the given axes.
    """

    # Choose functions based on target axis
    if axis == 'x':
        axScale = ax.get_xscale()
        lim = ax.get_xlim()
        set_lim = ax.set_xlim
    elif axis == 'y':
        axScale = ax.get_yscale()
        lim = ax.get_ylim()
        set_lim = ax.set_ylim
    else:
        raise ValueError("Unrecognized ``axis`` = '%s'!!" % (str(axis)))

    lim = np.array(lim)

    # Determine axis scaling
    if axScale.startswith('lin'):
        log = False
    elif axScale.startswith('log'):
        log = True
    else:
        raise ValueError("``axScale`` '%s' not implemented!" % (str(axScale)))

    # Convert to log if appropriate
    if log:
        lim = np.log10(lim)
        loc = np.log10(loc)

    # Find new axis bounds
    delta = np.diff(zmath.minmax(lim))[0]
    lim = np.array([loc - (0.5/scale)*delta, loc + (0.5/scale)*delta])
    # Convert back to linear if appropriate
    if log: lim = np.power(10.0, lim)
    set_lim(lim)

    return lim


def stretchAxes(ax, xs=1.0, ys=1.0):
    """
    Stretch the `x` and/or `y` limits of the given axes by a scaling factor.
    """

    xlog = (ax.get_xscale() == 'log')
    ylog = (ax.get_yscale() == 'log')

    xlims = np.array(ax.get_xlim())
    ylims = np.array(ax.get_ylim())

    if xlog: xlims = np.log10(xlims)
    if ylog: ylims = np.log10(ylims)

    xlims = [xlims[0] + 0.5*(1.0-xs)*(xlims[1]-xlims[0]),
             xlims[1] + 0.5*(1.0-xs)*(xlims[0]-xlims[1])]

    ylims = [ylims[0] + 0.5*(1.0-ys)*(ylims[1]-ylims[0]),
             ylims[1] + 0.5*(1.0-ys)*(ylims[0]-ylims[1])]

    if xlog: xlims = np.power(10.0, xlims)
    if ylog: ylims = np.power(10.0, ylims)

    ax.set_xlim(xlims)
    ax.set_ylim(ylims)

    return ax


def transform(ax, trans, fig=None):
    """Create a (blended) transformation.
    """
    if np.size(trans) == 1 and isinstance(trans, six.string_types):
        trans = [trans]
    elif np.size(trans) != 2:
        raise ValueError("`trans` = '{}' must be a string or pair of strings.".format(trans))

    forms = []
    for tt in trans:
        if tt.startswith('ax'):
            forms.append(ax.transAxes)
        elif tt.startswith('data'):
            forms.append(ax.transData)
        elif tt.startswith('fig'):
            if fig is None:
                raise ValueError("trans = '{}' requires a `fig` object".format(tt))
            forms.append(fig.transFigure)
        else:
            raise ValueError("Unrecognized transform '{}'".format(tt))

    if len(forms) == 1:
        transform = forms[0]
    else:
        transform = mpl.transforms.blended_transform_factory(forms[0], forms[1])

    return transform


def text(art, pstr, loc=None, x=None, y=None, halign=None, valign=None,
         fs=16, trans=None, pad=None, shift=None, **kwargs):
    """Add text to figure.

    Wrapper for the `matplotlib.figure.Figure.text` method.

    Arguments
    ---------
    art : `matplotlib.figure.Figure` or `matplotlib.axes.Axes` object,
    pstr : str,
        String to be printed.
    loc : str,
        String with two letters specifying the horizontal and vertical positioning of the text.
    x : float,
        X-position at which to draw the string, relative to the transformation given by `trans`.
    y : float,
        Y-position at which to draw the string, relative to the transformation given by `trans`.
    halign : str, one of {'center', 'left', 'right'},
        Horizontal alignment of text.
    valign : str, one of {'center', 'bottom', 'top'},
        Vertical alignment of text.
    fs : int,
        Fontsize.
    trans : `matplotlib.BboxTransformTo` object, or `None`,
        Transformation to use for text placement.
    pad : scalar or `None`

    shift : (2,) scalar or `None`
        Adjust the (x,y) position of the text by this amount.
    kwargs : any,
        Additional named arguments passed to `matplotlib.figure.Figure.text`.
        For example, ``color='blue'``, or ``rotation=90``.

    Returns
    -------
    txt : ``matplotlib.text.Text`` object,
        Handle storing the drawn text.

    """
    # if trans is None: trans = fig.transFigure
    if trans is None:
        if isinstance(art, mpl.figure.Figure):
            trans = art.transFigure
        elif isinstance(art, mpl.axes.Axes):
            trans = art.transAxes

    if pad is None:
        pad = _PAD

    # If a location string is given, convert to parameters
    if loc is not None:
        x, y, halign, valign = _loc_str_to_pars(
            loc, x=x, y=y, halign=halign, valign=valign, pad=pad)

    # Set default values
    if x is None:
        x = 0.5
    if y is None:
        y = 1 - pad

    if shift is not None:
        x += shift[0]
        y += shift[1]

    halign, valign = _parse_align(halign, valign)
    txt = art.text(x, y, pstr, size=fs, transform=trans,
                   horizontalalignment=halign, verticalalignment=valign, **kwargs)

    return txt


def label_line(ax, line, label, color='0.5', fs=14, halign='left', scale='linear', clip_on=True,
               halign_scale=1.0):
    """Add an annotation to the given line with appropriate placement and rotation.

    Based on code from:
        [How to rotate matplotlib annotation to match a line?]
        (http://stackoverflow.com/a/18800233/230468)
        User: [Adam](http://stackoverflow.com/users/321772/adam)

    Arguments
    ---------
    ax : `matplotlib.axes.Axes` object
        Axes on which the label should be added.
    line : `matplotlib.lines.Line2D` object
        Line which is being labeled.
    label : str
        Text which should be drawn as the label.
    ...

    Returns
    -------
    text : `matplotlib.text.Text` object

    """
    xlim = np.array(ax.get_xlim())
    ylim = np.array(ax.get_ylim())

    xdata, ydata = line.get_data()
    x1 = xdata[0]
    x2 = xdata[-1]
    y1 = ydata[0]
    y2 = ydata[-1]
    # Limit the edges to the plotted area
    x1, x2 = zmath.limit([x1, x2], xlim)
    y1, y2 = np.interp([x1, x2], xdata, ydata)
    y1, y2 = zmath.limit([y1, y2], ylim)
    x1, x2 = np.interp([y1, y2], ydata, xdata)

    log = _scale_to_log_flag(scale)

    if halign.startswith('l'):
        xx = x1*halign_scale
        halign = 'left'
    elif halign.startswith('r'):
        xx = halign_scale*x2
        halign = 'right'
    elif halign.startswith('c'):
        xx = zmath.midpoints([x1, x2], log=log)*halign_scale
        halign = 'center'
    else:
        raise ValueError("Unrecognized `halign` = '{}'.".format(halign))

    yy = np.interp(xx, xdata, ydata)

    # Add Annotation to Text
    xytext = (0, 0)
    text = ax.annotate(label, xy=(xx, yy), xytext=xytext, textcoords='offset points',
                       size=fs, color=color, zorder=1, clip_on=clip_on,
                       horizontalalignment=halign, verticalalignment='center_baseline')
    sp1 = ax.transData.transform_point((x1, y1))
    sp2 = ax.transData.transform_point((x2, y2))

    rise = (sp2[1] - sp1[1])
    run = (sp2[0] - sp1[0])

    slope_degrees = np.degrees(np.arctan2(rise, run))
    text.set_rotation_mode('anchor')
    text.set_rotation(slope_degrees)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    return text


def legend(art, keys, names, x=None, y=None, halign='right', valign='center', fs=12, trans=None,
           fs_title=None, loc=None, mono=False, zorder=None, **kwargs):
    """Add a legend to the given figure.

    Wrapper for the `matplotlib.pyplot.Legend` method.

    Arguments
    ---------
    art : `matplotlib.figure.Figure` pr `matplotlib.axes.Axes` object,
    keys : array_like of artists, shape (N,)
        Handles to the legend artists to be included in the legend.
    names : array_like of str, shape (N,)
        Names corresponding to each legend artist in `keys`.
    x : float,
        X-position at which to draw the legend, relative to the transformation given by `trans`.
    y : float,
        Y-position at which to draw the legend, relative to the transformation given by `trans`.
    halign : str, one of {'center', 'left', 'right'},
        Horizontal alignment of legend box.
    valign : str, one of {'center', 'lower', 'upper'},
        Vertical alignment of legend box.
    fs : int,
        Fontsize.
    trans : `matplotlib.BboxTransformTo` object, or `None`,
        Transformation to use for legend placement.
        If `None`, then it defaults to `transFigure` or `transAxes` if `art` is a 'Figure' or 'Axes'
        respectively.
    fs_title : int,
    loc : str or 'None',
        Describe the location of the legend using a string, e.g. 'tl', 'br', 'cl', 'tc'
        The string must be a two letter combination, such that:
        -   First letter determines the vertical alingment {'t', 'b', 'c'};
        -   Second letter the horizontal, {'l', 'r', 'c'}.
    mono : bool,
        Use a monospace font for the legend strings.
    kwargs : any,
        Additional named arguments passed to `matplotlib.pyplot.legend`.
        For example, ``ncol=1`` or ``title='Legend Title'``.

    Returns
    -------
    leg : ``matplotlib.legend.Legend`` object,
        Handle storing the drawn legend.

    """
    if isinstance(art, mpl.figure.Figure):
        ax = art.axes[0]
        if trans is None:
            trans = art.transFigure
    elif isinstance(art, mpl.axes.Axes):
        ax = art
        if trans is None:
            trans = ax.transAxes

    if 'handlelength' not in kwargs:
        kwargs['handlelength'] = _HANDLE_LENGTH
    if 'scatterpoints' not in kwargs:
        kwargs['scatterpoints'] = _SCATTER_POINTS
    # `alpha` should actually be `framealpha`
    if 'alpha' in kwargs:
        warnings.warn("For legends, use `framealpha` instead of `alpha`.")
        kwargs['framealpha'] = kwargs.pop('alpha')
        # del kwargs['alpha']

    # Override alignment using `loc` argument
    if loc is not None:
        x, y, halign, valign = _loc_str_to_pars(loc)

    if valign == 'top':
        valign = 'upper'
    if valign == 'bottom':
        valign = 'lower'

    if x is None: x = 0.99
    if y is None: y = 0.5

    alignStr = valign
    if not (valign == 'center' and halign == 'center'):
        alignStr += " " + halign

    prop_dict = {'size': fs}
    if mono:
        prop_dict['family'] = 'monospace'
    leg = ax.legend(keys, names, prop=prop_dict,
                    loc=alignStr, bbox_transform=trans, bbox_to_anchor=(x, y), **kwargs)
    if fs_title is not None:
        plt.setp(leg.get_title(), fontsize=fs_title)

    if zorder is not None:
        leg.set_zorder(10)

    return leg


def unifyAxesLimits(axes, axis='y'):
    """Given a list of axes, set all limits to match global extrema.
    """

    assert axis in ['x', 'y'], "``axis`` must be either 'x' or 'y' !!"

    if axis == 'y':
        lims = np.array([ax.get_ylim() for ax in axes])
    else:
        lims = np.array([ax.get_xlim() for ax in axes])

    lo = np.min(lims[:, 0])
    hi = np.max(lims[:, 1])

    for ax in axes:
        if axis == 'y':
            ax.set_ylim([lo, hi])
        else:
            ax.set_xlim([lo, hi])

    return np.array([lo, hi])


def colorCycle(args, **kwargs):
    """Create a range of colors.  DEPRECATED: use `color_cycle`.
    """
    warnings.warn("Use `color_cycle` function.", DeprecationWarning, stacklevel=3)
    return color_cycle(args, **kwargs)


def color_cycle(num, ax=None, color=None, cmap=plt.cm.spectral, left=0.1, right=0.9, light=True):
    """Create a range of colors.

    Arguments
    ---------
    num : int
        Number of colors to put in cycle.
    ax : ``matplotlib.axes.Axes`` object or `None`
        Axes on which to set the colors.  If given, then subsequent calls to ``ax.plot`` will use
        the different colors of the color-cycle.  If `None`, then the created colorcycle is only
        returned.
    cmap : ``matplotlib.colors.Colormap`` object
        Colormap from which to select colors.
    left : float {0.0, 1.0}
        Start colors this fraction of the way into the colormap (to avoid black/white).
    right : float {0.0, 1.0}
        Stop colors at this fraction of the way through the colormap (to avoid black/white).
    light : bool
        If `color` is given instead of `cmap`, use a seaborn 'light' colormap (vs. 'dark').
        Note: only works if `color` is given.

    Returns
    -------
    cols : (`num`,) array_like of RGBA color tuples
        Colors forming the color cycle.

    """
    nums = np.linspace(left, right, num)

    # If a single color is not provided, use a colormap (`cmap`)
    if color is None:
        cmap = _get_cmap(cmap)
    # If a single color is provided, create a cycle by altering its `a[lpha]`
    else:
        if isinstance(color, six.string_types):
            cc = mpl.colors.ColorConverter()
            color = cc.to_rgba(color)
        if np.size(color) == 3:
            color = np.append(color, 1.0)
        if np.size(color) != 4:
            raise ValueError("`color` = '{}', must be a RGBA series.".format(color))

        if light:
            palette = sns.light_palette
        else:
            palette = sns.dark_palette

        cmap = palette(color, n_colors=num, as_cmap=True)

    cols = [cmap(it) for it in nums]
    if ax is not None:
        ax.set_color_cycle(cols[::-1])
    return cols


def colormap(args, cmap=None, scale=None, under='0.8', over='0.8', left=None, right=None):
    """Create a colormap from a scalar range to a set of colors.

    Arguments
    ---------
    args : scalar or array_like of scalar
        Range of valid scalar values to normalize with
    cmap : ``matplotlib.colors.Colormap`` object
        Colormap to use.
    scale : str or `None`
        Scaling specification of colormap {'lin', 'log', `None`}.
        If `None`, scaling is inferred based on input `args`.
    under : str or `None`
        Color specification for values below range.
    over : str or `None`
        Color specification for values above range.
    left : float {0.0, 1.0} or `None`
        Truncate the left edge of the colormap to this value.
        If `None`, 0.0 used (if `right` is provided).
    right : float {0.0, 1.0} or `None`
        Truncate the right edge of the colormap to this value
        If `None`, 1.0 used (if `left` is provided).

    Returns
    -------
    smap : ``matplotlib.cm.ScalarMappable``
        Scalar mappable object which contains the members:
        `norm`, `cmap`, and the function `to_rgba`.

    Notes
    -----
    -   Truncation:
        -   If neither `left` nor `right` is given, no truncation is performed.
        -   If only one is given, the other is set to the extreme value: 0.0 or 1.0.

    """

    if cmap is None:
        cmap = 'jet'
    if isinstance(cmap, six.string_types):
        cmap = plt.get_cmap(cmap)

    # Select a truncated subsection of the colormap
    if (left is not None) or (right is not None):
        if left is None:
            left = 0.0
        if right is None:
            right = 1.0
        cmap = cut_colormap(cmap, left, right)

    if under is not None:
        cmap.set_under(under)
    if over is not None:
        cmap.set_over(over)

    if scale is None:
        if np.size(args) > 1 and np.all(args > 0.0):
            scale = 'log'
        else:
            scale = 'lin'

    log = _scale_to_log_flag(scale)
    if log:
        filter = 'g'
    else:
        filter = None

    # Determine minimum and maximum
    if np.size(args) > 1:
        min, max = zmath.minmax(args, filter=filter)
    else:
        min, max = 0, np.int(args)-1

    # Create normalization
    if log:
        norm = mpl.colors.LogNorm(vmin=min, vmax=max)
    else:
        norm = mpl.colors.Normalize(vmin=min, vmax=max)

    # Create scalar-mappable
    smap = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    # Bug-Fix something something
    smap._A = []
    # Store type of mapping
    smap.log = log

    return smap


def cut_colormap(cmap, min=0.0, max=1.0, n=100):
    """Select a truncated subset of the given colormap.

    Code from: http://stackoverflow.com/a/18926541/230468

    Arguments
    ---------
    cmap : `matplotlib.colors.Colormap`
        Colormap to truncate
    min : float, {0.0, 1.0}
        Minimum edge of the colormap
    max : float, {0.0, 1.0}
        Maximum edge of the colormap
    n : int
        Number of points to use for sampling

    Returns
    -------
    new_cmap : `matplotlib.colors.Colormap`
        Truncated colormap.

    """
    name = 'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=min, b=max)
    new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        name, cmap(np.linspace(min, max, n)))
    return new_cmap


def color_set(num, black=False, cset='xkcd'):
    """Retrieve a (small) set of color-strings with hand picked values.

    Arguments
    ---------
    num : int
        Number of colors to retrieve.
    black : bool
        Include 'black' as the first color.
    cset : str, {'xkcd', 'def'}
        Which set of colors to choose from.

    Returns
    -------
    cols : (`num`) list of str or RGBA tuples
        List of `matplotlib` compatible color-strings or tuples.

    """
    if cset == 'xkcd':
        colors = list(_COLOR_SET_XKCD)
        colors = sns.xkcd_palette(colors)
    elif cset.startswith('def'):
        colors = list(_COLOR_SET)
    else:
        raise ValueError("`cset` '{}' unrecognized.".format(cset))

    if black:
        colors = ['black'] + colors

    ncol = len(colors)
    # If more colors are requested than are available, fallback to `color_cycle`
    if num > ncol:
        # raise ValueError("Limited to {} colors, cannot produce `num` = '{}'.".format(ncol, num))
        colors = color_cycle(num)
        return colors

    return colors[:num]


def line_style_set(num):
    """Retrieve a (small) set of line-style specifications with hand constructed patterns.

    Used by the `matplotlib.lines.Line2D.set_dashes` method.
    The first element is a solid line.

    Arguments
    ---------
    num : int
        Number of line-styles to retrieve.

    Returns
    -------
    lines : (`num`) list of tuples,
        Set of line-styles.  Each line style is a tuple of values specifying dash spacings.

    """
    lines = list(_LINE_STYLE_SET)
    nline = len(lines)
    # If more colors are requested than are available, fallback to `color_cycle`
    if num > nline:
        raise ValueError("Limited to {} line-styles.".format(nline))

    return lines[:num]


def setGrid(*args, **kwargs):
    utils.dep_warn("setGrid", newname="set_grid")
    return set_grid(*args, **kwargs)


def set_grid(ax, val=True, axis='both', c=None, ls='-', clear=True,
             below=True, major=True, minor=True, zorder=2, alpha=None):
    """Configure the axes' grid.
    """
    if clear:
        ax.grid(False, which='both', axis='both')
    ax.set_axisbelow(below)
    if val:
        if major:
            if c is None:
                _c = '0.60'
            else:
                _c = c
            if alpha is None:
                _alpha = 0.8
            else:
                _alpha = alpha
            ax.grid(True, which='major', axis=axis, c=_c, ls=ls, zorder=zorder, alpha=_alpha)
        if minor:
            if c is None:
                _c = '0.85'
            else:
                _c = c
            if alpha is None:
                _alpha = 0.5
            else:
                _alpha = alpha
            ax.grid(True, which='minor', axis=axis, c=_c, ls=ls, zorder=zorder, alpha=_alpha)
    return


def skipTicks(ax, axis='y', skip=2, num=None, first=None, last=None):
    """
    Only label every ``skip`` tick marks.

    Arguments
    ---------
        ax    <obj>  : `matplotlib.axes.Axes` object, base axes class
        axis  <str>  : which axis to modify
        skip  <int>  : interval which to skip
        num   <int>  : target number of tick labels (``None`` : used a fixed ``skip``)
        first <bool> : If `True` always show first tick, if `False` never show, otherwise use skip
        last  <bool> : If `True` always show last  tick, if `False` never show, otherwise use skip

    """

    # Get the correct labels
    if axis == 'y': ax_labels = ax.yaxis.get_ticklabels()
    elif axis == 'x': ax_labels = ax.yaxis.get_ticklabels()
    else: raise RuntimeError("Unrecognized ``axis`` = '%s'!!" % (axis))

    count = len(ax_labels)

    # Determine ``skip`` to match target number of labels
    if num is not None: skip = np.int(np.ceil(1.0*count/num))

    visible = np.zeros(count, dtype=bool)

    # Choose some to be visible
    visible[::skip] = True

    if first is True: visible[0] = True
    elif first is False: visible[0] = False

    if last is True: visible[-1] = True
    elif last is False: visible[-1] = False

    for label, vis in zip(ax_labels, visible): label.set_visible(vis)

    return


def saveFigure(fig, fname, verbose=True, log=None, level=logging.WARNING, close=True, **kwargs):
    """Save the given figure(s) to the given filename.

    If ``fig`` is iterable, a multipage pdf is created.  Otherwise a single file is made.
    Does *not* make sure path exists.

    Arguments
    ---------
        fig      <obj>([N]) : one or multiple ``matplotlib.figure.Figure`` objects.
        fname    <str>      : filename to save to.

        verbose  <bool>     : print verbose output to stdout
        log      <obj>      : ``logging.Logger`` object to print output to
        level    <int>      :
        close    <bool>     : close figures after saving
        **kwargs <dict>     : additional arguments past to ``savefig()``.
    """

    # CATCH WRONG ORDER OF ARGUMENTS
    if type(fig) == str:
        warnings.warn("FIRST ARGUMENT SHOULD BE `fig`!!")
        temp = str(fig)
        fig = fname
        fname = temp

    if log is not None: log.debug("Saving figure...")

    if not np.iterable(fig): fig = [fig]
    saved_names = []

    # Save as multipage PDF
    if fname.endswith('pdf') and np.size(fig) > 1:
        from matplotlib.backends.backend_pdf import PdfPages
        with PdfPages(fname) as pdf:
            for ff in fig:
                pdf.savefig(figure=ff, **kwargs)
                if close: plt.close(ff)
                # Make sure file now exists
                if os.path.exists(fname):
                    saved_names.append(fname)
                else:
                    raise RuntimeError("Figure '{}' did not save.".format(fname))

    else:
        # Save each figure to a different file
        for ii, ff in enumerate(fig):
            # On subsequent figures, append the number to the filename
            if ii == 0:
                usefname = str(fname)
            else:
                usefname = zio.modify_filename(fname, append='_%d' % (ii))

            ff.savefig(usefname, **kwargs)
            if close: plt.close(ff)
            if os.path.exists(usefname):
                saved_names.append(usefname)
            else:
                raise RuntimeError("Figure '{}' did not save.".format(usefname))

    # No files saved or Some files were not saved
    if not len(saved_names) or len(saved_names) != len(fig):
        warn_str = "Error saving figures..."
        if log is None:
            warnings.warn(warn_str)
        else:
            log.warning(warn_str)

    # Things look good.
    else:
        printStr = "Saved figure to '%s'" % (fname)
        if log is not None:
            log.log(level, printStr)
        elif verbose:
            print(printStr)

    return


def strSciNot(val, precman=0, precexp=0, dollar=True, one=True, zero=False):
    """Convert a scalar into a string with scientific notation (latex formatted).

    Arguments
    ---------
    val : scalar
        Numerical value to convert.
    precman : int or `None`
        Precision of the mantissa (decimal points); or `None` for omit mantissa.
    precexp : int or `None`
        Precision of the exponent (decimal points); or `None` for omit exponent.
    dollar : bool
        Include dollar-signs ('$') around returned expression.
    one : bool
        Include the mantissa even if it is '1[.0...]'.
    zero : bool
        If the value is uniformly '0.0', write it as such (instead of 10^-inf).

    Returns
    -------
    notStr : str
        Scientific notation string using latex formatting.

    """
    if zero and val == 0.0:
        notStr = "$"*dollar + "0.0" + "$"*dollar
        return notStr

    man, exp = zmath.frexp10(val)
    use_man = (precman is not None and np.isfinite(exp))
    if use_man: manStr = "{0:.{1:d}f}".format(man, precman)
    else:       manStr = ""
    # If the mantissa is '1' (or '1.0' or '1.00' etc), dont write it
    if not one and manStr == "{0:.{1:d}f}".format(1.0, precman):
        manStr = ""

    if precexp is not None:
        # Try to convert `exp` to integer, fails if 'inf' or 'nan'
        try:
            exp = np.int(exp)
            expStr = "10^{{ {:d} }}".format(exp)
        except:
            expStr = "10^{{ {0:.{1:d}f} }}".format(exp, precexp)

        # Add negative sign if needed
        if not use_man and (man < 0.0 or val == -np.inf):
            expStr = "-" + expStr
    else:
        expStr = ""

    notStr = "$"*dollar + manStr
    if len(manStr) and len(expStr):
        notStr += " \\times"
    notStr += expStr + "$"*dollar

    return notStr


def plotHistLine(ax, edges, hist, yerr=None, nonzero=False, positive=False, extend=None,
                 fill=None, invert=False, **kwargs):
    """Given bin edges and histogram-like values, plot a histogram.

    Arguments
    ---------
        ax       <obj>    : matplotlib axes object on which to plot
        edges    <flt>[N] : positions of bin edges, length either that of hist ``M`` or ``M+1``
        hist     <flt>[M] : histogram values for each bin
        yerr     <flt>[M] : value for y-error-bars
        nonzero  <bool>   : only plot non-zero values
        positive <bool>   : only plot positive values
        extend   <str>    : required if ``N != M+1``, sets direction to extend ``edges``
        fill     <obj>    : If not ``None``, fill below line; can set as dict of fill parameters
        **kwargs <dict>   : key value pairs to be passed to the plotting function

    Returns
    -------
        line     <obj>    : the plotted object

    """

    # Determine color if included in ``kwargs``
    col = 'black'
    if kwargs.get('color') is not None:
        col = kwargs.get('color')
    elif kwargs.get('c') is not None:
        col = kwargs.get('c')

    # Extend bin edges if needed
    if len(edges) != len(hist)+1:
        if extend == 'left':
            edges = np.concatenate([[zmath.extend(edges)[0]], edges])
        elif extend == 'right':
            edges = np.concatenate([edges, [zmath.extend(edges)[1]]])
        else:
            raise RuntimeError("``edges`` must be longer than ``hist``, or ``extend`` given")

    # Construct plot points to manually create a step-plot
    xval, yval = _histLine(edges, hist)

    # Select nonzero values
    if nonzero:
        xval = np.ma.masked_where(yval == 0.0, xval)
        yval = np.ma.masked_where(yval == 0.0, yval)

    # Select positive values
    if positive:
        xval = np.ma.masked_where(yval < 0.0, xval)
        yval = np.ma.masked_where(yval < 0.0, yval)

    if invert:
        temp = np.array(xval)
        xval = yval
        yval = temp

    # Plot Histogram
    line, = ax.plot(xval, yval, **kwargs)

    # Add yerror-bars
    if yerr is not None:
        xmid = zmath.midpoints(edges)

        if nonzero:
            inds = np.where(hist != 0.0)
            ax.errorbar(xmid[inds], hist[inds], yerr=yerr[inds], fmt=None, ecolor=col)
        else:
            ax.errorbar(xmid,       hist,       yerr=yerr,       fmt=None, ecolor=col)

    # Add a fill region
    if fill is not None:
        ylim = ax.get_ylim()
        if type(fill) == dict:
            filldict = fill
        else:
            filldict = dict()
        ax.fill_between(xval, yval, 0.1*ylim[0], **filldict)
        ax.set_ylim(ylim)

    return line


def plotSegmentedLine(ax, xx, yy, zz=None, cmap=plt.cm.jet, norm=[0.0, 1.0], lw=3.0, alpha=1.0):
    """Draw a line segment by segment.
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb

    Plot a colored line with coordinates xx and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """

    # Get the minimum and maximum of ``norm``
    norm = zmath.minmax(norm)
    # conver to normalization
    norm = plt.Normalize(norm[0], norm[1])

    if zz is None: zz = np.linspace(norm.vmin, norm.vmax, num=len(xx))
    else:             zz = np.asarray(zz)

    segments = _make_segments(xx, yy)
    lc = mpl.collections.LineCollection(segments, array=zz, cmap=cmap, norm=norm,
                                        linewidth=lw, alpha=alpha)

    ax.add_collection(lc)

    return lc


def plotScatter(ax, xx, yy, scalex='log', scaley='log',
                size=None, cont=False, color=None, alpha=None, **kwargs):
    """Draw a scatter plot.
    """
    COL = COL_CORR

    if size is None: size = [30, 6]
    if color is None: color = ['0.25', COL]
    if alpha is None: alpha = [0.5, 0.8]

    ax.scatter(xx, yy, s=size[0], color=color[0], alpha=alpha[0], **kwargs)
    pnts = ax.scatter(xx, yy, s=size[1], color=color[1], alpha=alpha[1], **kwargs)

    # Add Contours
    if cont:
        NUM = 4
        CMAP = 'jet'
        res = 0.3*np.sqrt(len(xx))
        res = np.max([1, res])

        smap = colormap(NUM, CMAP, scale='lin')
        cols = [smap.to_rgba(it) for it in range(NUM)]

        xi = zmath.spacing(xx, scalex, num=res)
        yi = zmath.spacing(yy, scaley, num=res)
        hist, xedges, yedges = np.histogram2d(xx, yy, (xi, yi))
        gx, gy = np.meshgrid(xedges[1:], yedges[1:])
        ax.contour(gx, gy, hist.T, NUM, colors=cols)

    return pnts


def plotHistBars(ax, xx, bins=20, scalex='log', scaley='log', conf=True, **kwargs):
    """Plot a histogram bar graph.

    NOTE: For log-y, make sure `yscale` is either not manually set, or include `nonposy='clip'`

    Arguments
    ---------
    ax : ``matplotlib.axes.Axes`` object,
        Axes on which to plot.
    xx : (N,) array_like scalars,
        Values to be histogrammed.
    bins : int or array_like,
        Either the number of bins for bin-edges to be automatically generated, or the bin-edges
        themselves.
    ...

    """
    HIST_ALPHA = 0.75
    CONF_ALPHA = 0.5

    CONF_INTS = [0.95, 0.68]
    CONF_COLS = ['green', 'orange']

    if scaley.startswith('log'): logy = True
    else:                        logy = False

    if 'color' not in kwargs and 'c' not in kwargs:
        kwargs['color'] = COL_CORR
    if 'alpha' not in kwargs:
        kwargs['alpha'] = HIST_ALPHA
    if 'rwidth' not in kwargs:
        kwargs['rwidth'] = 0.8
    if 'zorder' not in kwargs:
        kwargs['zorder'] = 100

    # Add Confidence intervals
    if conf:
        med, cints = zmath.confidenceIntervals(xx, ci=CONF_INTS)
        ax.axvline(med, color='red', ls='--', lw=LW_CONF, zorder=101)
        # Add average
        ave = np.average(xx)
        ax.axvline(ave, color='red', ls=':', lw=LW_CONF, zorder=101)
        for ii, (vals, col) in enumerate(zip(cints, CONF_COLS)):
            ax.axvspan(*vals, color=col, alpha=CONF_ALPHA)

    # Create a given number of log-spaced bins
    #     If not log-spaced, then `ax.hist` will do the same
    if isinstance(bins, numbers.Integral) and scalex.startswith('log'):
        bins = zmath.spacing(xx, num=bins, scale='log')

    cnts, bins, bars = ax.hist(xx, bins, histtype='bar', log=logy, **kwargs)

    # Dont let lower y-lim be less than 0.8 with log-scaling
    if scaley.startswith('log'):
        # setLim(ax, 'y', lo=0.8, at='least')   <=== This isn't working for some reason!  FIX
        ylim = np.array(ax.get_ylim())
        if ylim[0] < 0.8:
            ylim[0] = 0.8
            ax.set_ylim(ylim)

    return bars


def plotConfFill(ax, rads, med, conf, color='red', fillalpha=0.5, lw=1.0, linealpha=0.8,
                 filter=None, outline='0.5', edges=True, floor=None, ceil=None, dashes=None,
                 lw_edges=None,
                 **kwargs):
    """Draw a median line and (set of) confidence interval(s).

    The `med` and `conf` values can be obtained from `numpy.percentile` and or
    `zcode.math.confidenceIntervals`.

    Arguments
    ---------
    ax : `matplotlib.axes.Axes` object,
        Axes on which to plot.
    rads : (N,) array_like scalars,
        Radii corresponding to each median and confidence value.
    med : (N,) array_like scalars,
        Median values to plot as line.
    conf : (N,[M,]2,) array_like scalars,
        Confidence intervals to plot as shaded regions.  There can be `M` different confidence
        intervals, each with an upper and lower value.  Shape of `(N,2,)` is also allowed for a
        single confidence interval.
    color : `matplotlib` color spec,
        Color for median lines and shaded region.
    fillalpha : float or (M,) array_like of floats,
        Alpha (opacity) specification for fill regions; each element must be ``{0.0, 1.0}``.
        If ``(M,)`` are given, one is used for each confidence-interval.  Otherwise, for
        confidence interval `i`, the alpha used is ``fillalpha^{i+1}``.
    lw : float,
        Line-weight used specifically for the median line (not the filled-regions).
    filter : str or `None`
        Apply a relative-to-zero filter to the y-data before plotting.
    outline : str or `None`.
        Draw a `outline` colored line behind the median line to make it easier to see.
        If `None`, no outline is drawn.
    edges : bool
        Add lines along the edges of each confidence interval.
    floor : array_like of float or `None`
        Set the minimum value for confidence intervals to this value.
    ceil : array_like of float or `None`
        Set the maximmum value for confidence intervals to be this value.
    dashes :
    **kwargs : additional key-value pairs,
        Passed to `matplotlib.pyplot.fill_between` controlling `matplotlib.patches.Polygon`
        properties.  These are included in the `linePatch` objects, but *not* the `confPatches`.

    Returns
    -------
    linePatch : `matplotlib.patches.Patch`,
        Composite patch of median line and shaded region patch (for use on legend).
    confPatches : (M,) list of `matplotlib.patches.Patch`,
        A patch for each confidence inteval (for use on legend).

    """
    conf = np.atleast_2d(conf)
    if conf.shape[-1] != 2:
        raise ValueError("Last dimension of `conf` must be 2!")

    # `conf` has shape ``(num-rads, num-conf-ints, 2)``
    if conf.ndim == 2:
        conf = conf.reshape(len(rads), 1, 2)
    elif conf.ndim != 3:
        raise ValueError("`conf` must be 2 or 3 dimensions!")

    if filter is not None:
        filter = zmath._comparisonFunction(filter)

    if lw_edges is None:
        lw_edges = 0.5 * lw

    numConf = np.shape(conf)[-2]
    confPatches = []
    # Iterate over confidence intervals
    _pp = None
    for jj in xrange(numConf):
        # Set fill-opacity
        if np.size(fillalpha) == numConf:
            falph = fillalpha
        else:
            falph = np.power(fillalpha, jj+1)

        xx = np.array(rads)
        ylo = np.array(conf[:, jj, 0])
        yhi = np.array(conf[:, jj, 1])
        if floor is not None:
            ylo = np.maximum(ylo, floor)
        if ceil is not None:
            yhi = np.minimum(yhi, ceil)

        if filter:
            ylo = np.ma.masked_where(~filter(ylo, 0.0), ylo)
            yhi = np.ma.masked_where(~filter(yhi, 0.0), yhi)

        # Fill between confidence intervals
        pp = ax.fill_between(xx, ylo, yhi, alpha=falph, facecolor=color, **kwargs)
        confPatches.append(pp)

        # Plot edges of confidence intervals
        if edges:
            ax.plot(rads, ylo, color=color, alpha=0.5*linealpha, lw=lw_edges)
            ax.plot(rads, yhi, color=color, alpha=0.5*linealpha, lw=lw_edges)

        # Create dummy-patch for the median-line and fill-color, for a legend
        if jj == 0:
            # pp = ax.fill(np.nan, np.nan, facecolor=color, alpha=falph, **kwargs)
            # Create overlay of lines and patches
            _pp = pp
            # linePatch = (_pp, ll)

    # Plot Median Line
    #    Plot black outline to improve contrast
    if outline is not None:
        oo, = ax.plot(rads, med, '-', color=outline, lw=2*lw, alpha=_LW_OUTLINE)
        if dashes is not None:
            oo.set_dashes(tuple(dashes))

    ll, = ax.plot(rads, med, '-', color=color, lw=lw, alpha=linealpha)
    if dashes is not None:
        ll.set_dashes(tuple(dashes))

    linePatch = (_pp, ll)

    return linePatch, confPatches


def line_label(ax, pos, label, dir='v', loc='top', xx=None, yy=None, ha=None, va=None,
               line_kwargs={}, text_kwargs={}, dashes=None, rot=None):
    """Plot a vertical line, and give it a label outside the axes.

    Arguments
    ---------
    ax : `matplotlib.axes.Axes` object
        Axes on which to plot.
    xx : float
        Location (in data coordinated) to place the line.
    label : str
        Label to place with the vertical line.
    top : bool
        Place the label above the axes ('True'), as apposed to below ('False').
    line_kwargs : dict
        Additional parameters for the line, passed to `ax.axvline`.
    text_kwargs : dict
        Additional parameters for the text, passed to `plot_core.text`.
    dashes : array_like of float or `None`
        Specification for dash/dots pattern for the line, passed to `set_dashes`.

    Returns
    -------
    ll : `matplotlib.lines.Line2D`
        Added line object.
    txt : `matplotlib.text.Text`
        Added text object.

    """
    tdir = dir.lower()[:1]
    if tdir.startswith('v'):   VERT = True
    elif tdir.startswith('h'): VERT = False
    else: raise ValueError("`dir` ('{}') must start with {{'v', 'h'}}".format(dir))
    tloc = loc.lower()[:1]
    valid_locs = ['t', 'b', 'l', 'r']
    if tloc not in valid_locs:
        raise ValueError("`loc` ('{}') must start with '{}'".format(loc, valid_locs))

    # Set default rotation
    if rot is None:
        rot = 0
        # If to 'l'eft or 'r'ight, rotate 90-degrees
        if tloc.startswith('l'): rot = 90
        elif tloc.startswith('r'): rot = -90

    # Set alignment
    if tloc.startswith('l'):
        _ha = 'right'
        _va = 'center'
    elif tloc.startswith('r'):
        _ha = 'left'
        _va = 'center'
    elif tloc.startswith('t'):
        _ha = 'center'
        _va = 'bottom'
    elif tloc.startswith('b'):
        _ha = 'center'
        _va = 'top'

    if ha is None: ha = _ha
    if va is None: va = _va

    # Add vertical line
    if VERT:
        ll = ax.axvline(pos, **line_kwargs)
        trans = mpl.transforms.blended_transform_factory(ax.transData, ax.transAxes)
        if tloc.startswith('l'):
            _xx = pos
            _yy = 0.5
        elif tloc.startswith('r'):
            _xx = pos
            _yy = 0.5
        elif tloc.startswith('t'):
            _xx = pos
            _yy = 1.0 + _PAD
        elif tloc.startswith('b'):
            _xx = pos
            _yy = 0.0 - _PAD
    # Add horizontal line
    else:
        ll = ax.axhline(pos, **line_kwargs)
        trans = mpl.transforms.blended_transform_factory(ax.transAxes, ax.transData)
        if tloc.startswith('l'):
            _xx = 0.0 - _PAD
            _yy = pos
        elif tloc.startswith('r'):
            _xx = 1.0 + _PAD
            _yy = pos
        elif tloc.startswith('t'):
            _xx = 0.5
            _yy = pos
        elif tloc.startswith('b'):
            _xx = 0.5
            _yy = pos

    if xx is None: xx = _xx
    if yy is None: yy = _yy

    if dashes: ll.set_dashes(dashes)

    txt = text(ax, label, x=xx, y=yy, halign=ha, valign=va, trans=trans, **text_kwargs)
    return ll, txt


def full_extent(ax, pad=0.0, invert=None):
    """Get the full extent of an axes, including axes labels, tick labels, and titles.

    From: 'stackoverflow.com/questions/14712665/'
    """
    # Draw text objects so extents are defined
    ax.figure.canvas.draw()
    if isinstance(ax, mpl.axes.Axes):
        items = ax.get_xticklabels() + ax.get_yticklabels()
        items += [ax, ax.title, ax.xaxis.label, ax.yaxis.label]
        # items += [ax, ax.title]
        use_items = []
        for item in items:
            if isinstance(item, mpl.text.Text) and len(item.get_text()) == 0: continue
            use_items.append(item)
        # bbox = mpl.transforms.Bbox.union([item.get_window_extent() for item in use_items])
        bbox = _set_extents(use_items, union=True, pad=0.0)
    elif isinstance(ax, mpl.legend.Legend):
        bbox = ax.get_frame().get_bbox()
    else:
        err_str = "Unrecognized type of `ax` = '{}'.  Currently support axes and legends.".format(
            type(ax))
        raise ValueError(err_str)

    bbox = bbox.expanded(1.0 + pad, 1.0 + pad)
    if invert:
        bbox = bbox.transformed(invert.inverted())

    return bbox


def position_to_extent(fig, ref, loc, item=None, pad=0.0, halign='left', valign='lower'):
    """Reposition axis so that origin of 'full_extent' is at given `loc`.
    """
    if item is None:
        item = ref

    bbox = full_extent(ref, pad=pad, invert=fig.transFigure)
    ax_bbox = ref.get_position()

    if halign.startswith('r'):
        dx = ax_bbox.x1 - bbox.x1
    elif halign.startswith('l'):
        dx = ax_bbox.x0 - bbox.x0
    else:
        raise ValueError("`halign` = '{}' must start with 'l' or 'r'.".format(halign))

    if valign.startswith('l'):
        dy = ax_bbox.y0 - bbox.y0
    elif valign.startswith('u'):
        dy = ax_bbox.y1 - bbox.y1
    else:
        raise ValueError("`valign` = '{}' must start with 'l' or 'u'.".format(valign))

    if len(loc) == 2:
        new_loc = [loc[0]+dx, loc[1]+dy, ax_bbox.width, ax_bbox.height]
    elif len(loc) == 4:
        new_loc = [loc[0]+dx, loc[1]+dy, loc[2], loc[3]]
    else:
        raise ValueError("`loc` must be 2 or 4 long: [x, y, (width, height)]")

    item.set_position(new_loc)
    return


def _extents(ax, pad=0.0, invert=None, group=False):
    """Get the extents of an axes, including axes labels, tick labels, and titles.

    Adapted From: 'stackoverflow.com/questions/14712665/'
    """
    # For text objects, we need to draw the figure first, otherwise the extents
    # are undefined.
    ax.figure.canvas.draw()
    if group:
        items = [ax, ax.title, ax.xaxis.label, ax.yaxis.label]
        bboxes = [_set_extents(ax.get_xticklabels(), pad=pad, union=True)]
        bboxes += [_set_extents(ax.get_yticklabels(), pad=pad, union=True)]
        bboxes += _set_extents(items, pad=pad, union=False)
    else:
        items = ax.get_xticklabels() + ax.get_yticklabels()
        items += [ax, ax.title, ax.xaxis.label, ax.yaxis.label]
        # bboxes = [it.get_window_extent().expanded(1.0 + pad, 1.0 + pad) for it in items]
        bboxes = _set_extents(items, pad=pad, union=False)

    if invert:
        bboxes = [bb.transformed(invert.inverted()) for bb in bboxes]

    return bboxes


def _set_extents(items, pad=0.0, union=False):
    bboxes = [it.get_window_extent().expanded(1.0 + pad, 1.0 + pad) for it in items]
    if union:
        bboxes = mpl.transforms.Bbox.union(bboxes)
    return bboxes


def backdrop(fig, obj, pad=0.0, union=False, group=False, draw=True, **kwargs):
    """Draw a rectangle behind the full extent of the given object.

    Example
    -------
    >>> zplot.backdrop(fig, ax, pad=0.02, union=True,
                       facecolor='white', edgecolor='none', zorder=5, alpha=0.8)
    """
    if union:
        bboxes = [full_extent(obj, pad=pad, invert=fig.transFigure)]
    else:
        bboxes = _extents(obj, pad=pad, invert=fig.transFigure, group=group)

    pats = []
    for bbox in bboxes:
        rect = mpl.patches.Rectangle([bbox.xmin, bbox.ymin], bbox.width, bbox.height,
                                     transform=fig.transFigure, **kwargs)
        if draw: fig.patches.append(rect)
        pats.append(rect)

    if len(pats) == 1: return pats[0]
    return pats


def _setAxis_scale(ax, axis, scale, thresh=None):
    if scale.startswith('lin'): scale = 'linear'
    if scale == 'symlog': thresh = 1.0
    if axis == 'x': ax.set_xscale(scale, linthreshx=thresh)
    elif axis == 'y': ax.set_yscale(scale, linthreshy=thresh)
    else: raise RuntimeError("Unrecognized ``axis`` = %s" % (axis))
    return


def _setAxis_label(ax, axis, label, fs=12, c='black'):
    if axis == 'x': ax.set_xlabel(label, size=fs, color=c)
    elif axis == 'y': ax.set_ylabel(label, size=fs, color=c)
    else: raise RuntimeError("Unrecognized ``axis`` = %s" % (axis))
    return


def _histLine(edges, hist):
    xval = np.hstack([[edges[jj], edges[jj+1]] for jj in range(len(edges)-1)])
    yval = np.hstack([[hh, hh] for hh in hist])
    return xval, yval


def _clear_frame(ax=None):
    # Taken from a post by Tony S Yu
    if ax is None: ax = plt.gca()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    for spine in ax.spines.values(): spine.set_visible(False)
    return


def _make_segments(x, y):
    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb

    Create list of line segments from x and y coordinates, in the correct format for LineCollection:
    an array of the form   [numlines, (points per line), 2 (x and y)] array
    """
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments


def _scale_to_log_flag(scale):
    # Check formatting of `scale` str
    scale = _clean_scale(scale)
    if scale.startswith('log'):
        log = True
    elif scale.startswith('lin'):
        log = False
    else:
        raise ValueError("Unrecognized `scale` '{}'; must start with 'log' or 'lin'".format(scale))
    return log


def _clean_scale(scale):
    """Cleanup a 'scaling' string to be matplotlib compatible.
    """
    scale = scale.lower()
    if scale.startswith('lin'):
        scale = 'linear'
    return scale


def _get_cmap(cmap):
    """Retrieve a colormap with the given name if it is not already a colormap.
    """
    if isinstance(cmap, six.string_types):
        return mpl.cm.get_cmap(cmap)
    elif isinstance(cmap, mpl.colors.Colormap):
        return cmap
    else:
        raise ValueError("`cmap` '{}' is not a valid colormap or colormap name".format(cmap))


def _loc_str_to_pars(loc, x=None, y=None, halign=None, valign=None, pad=_PAD):
    """Convert from a string location specification to the specifying parameters.

    If any of the specifying parameters: {x, y, halign, valign}, are 'None', they are set to
    default values.

    Returns
    -------
    x : float
    y : float
    halign : str
    valign : str

    """
    _valid_loc = [['t', 'u', 'b', 'l', 'c'], ['l', 'r', 'c']]
    for ii, (ll, vv) in enumerate(zip(loc, _valid_loc)):
        if ll not in vv:
            err = "Unrecognized `loc`[{}] = '{}' (`loc` = '{}').".format(ii, ll, loc)
            err += "\n\t`loc`[{}] must be one of '{}'".format(ii, vv)
            raise ValueError(err)

    if loc[0] == 't' or loc[0] == 'u':
        if valign is None:
            valign = 'top'
        if y is None:
            y = 1 - pad
    elif loc[0] == 'b' or loc[0] == 'l':
        if valign is None:
            valign = 'bottom'
        if y is None:
            y = pad
    elif loc[0] == 'c':
        if valign is None:
            valign = 'center'
        if y is None:
            y = 0.5

    if loc[1] == 'l':
        if halign is None:
            halign = 'left'
        if x is None:
            x = pad
    elif loc[1] == 'r':
        if halign is None:
            halign = 'right'
        if x is None:
            x = 1 - pad
    elif loc[1] == 'c':
        if halign is None:
            halign = 'center'
        if x is None:
            x = 0.5

    return x, y, halign, valign


def _parse_align(halign=None, valign=None):
    if halign is None:
        halign = 'center'
    if valign is None:
        valign = 'top'

    if halign.startswith('l'):
        halign = 'left'
    elif halign.startswith('c'):
        halign = 'center'
    elif halign.startswith('r'):
        halign = 'right'

    if valign.startswith('t'):
        valign = 'top'
    elif valign.startswith('c'):
        valign = 'center'
    elif valign.startswith('b'):
        valign = 'bottom'

    if valign == 'upper':
        warnings.warn("Use `'top'` not `'upper'`!")
        valign = 'top'

    if valign == 'lower':
        warnings.warn("Use `'bottom'` not `'lower'`!")
        valign = 'bottom'
    return halign, valign


'''
def rescale(ax, which='both'):
    """
    """

    if which == 'x':
        scaley = False
        scalex = True
    elif which == 'y':
        scaley = True
        scalex = True
    elif which == 'both':
        scaley = True
        scalex = True
    else:
        raise ValueError("Unrecognized ``which`` = '%s'" % str(which))


    # recompute the ax.dataLim
    ax.relim()
    # update ax.viewLim using the new dataLim
    ax.autoscale_view(tight=True, scaley=scaley, scalex=scalex)
    plt.draw()

    return ax

# } rescale()
'''


'''
def limits(xx, yy, xlims):
    """
    """
    inds = np.where((xx >= np.min(xlims)) & (xx <= np.max(xlims)))[0]
    return zmath.minmax(yy[inds])
'''
