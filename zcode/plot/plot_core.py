"""General plotting functions.

Functions
---------
-   setAxis              - Set many different axis properties at once.
-   twinAxis             - Easily create and set a new twin axis (like `twinx()` or `twiny()`)
-   setLim               - Set limits on an axis
-   zoom                 - Zoom-in at a certain location on the given axes.
-   stretchAxes          - Stretch the `x` and/or `y` limits of the given axes by a scaling factor.
-   text                 - Add text to figure.
-   legend               - Add a legend to the given figure.
-   unifyAxesLimits      - Set limits on all given axes to match global extrema.
-   color_cycle          - Create a range of colors.
-   colormap             - Create a colormap from scalars to colors.
-   color_set            - Retrieve a (small) set of color-strings with hand picked values.
-   setGrid              - Configure the axes' grid.
-   skipTicks            - skip some tick marks
-   saveFigure           - Save the given figure(s) to the given filename.
-   strSciNot            - Convert a scalar into a string with scientific notation.

-   plotHistLine         - Plot a histogram line.
-   plotSegmentedLine    - Draw a line segment by segment.
-   plotScatter          - Draw a scatter plot.
-   plotHistBars         - Plot a histogram bar graph.
-   plotConfFill         - Draw a median line and (set of) confidence interval(s).

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

import logging
import numbers
import warnings

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

import zcode.math as zmath
import zcode.inout as zio

__all__ = ['setAxis', 'twinAxis', 'setLim', 'zoom', 'stretchAxes', 'text', 'legend',
           'unifyAxesLimits', 'color_cycle',
           'colorCycle', 'colormap', 'color_set', 'setGrid', 'skipTicks', 'saveFigure', 'strSciNot',
           'plotHistLine', 'plotSegmentedLine', 'plotScatter', 'plotHistBars', 'plotConfFill']

COL_CORR = 'royalblue'
LW_CONF = 1.0
VALID_SIDES = [None, 'left', 'right', 'top', 'bottom']
_COLOR_SET = ['black', 'blue', 'red', 'green', 'purple',
              'orange', 'cyan', 'brown', 'gold', 'pink',
              'forestgreen', 'grey', 'olive', 'coral', 'yellow']
_LW_OUTLINE = 0.8


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

    assert axis in ['x', 'y'],                          "``axis`` must be `x` or `y`!"
    assert trans in ['axes', 'figure'],                  "``trans`` must be `axes` or `figure`!"
    assert side in VALID_SIDES, "``side`` must be in '%s'" % (VALID_SIDES)

    # Set tick colors and font-sizes
    ax.tick_params(axis=axis, which='both', colors=c, labelsize=fs)
    #    Set tick-size only for major ticks
    ax.tick_params(axis=axis, which='major', size=ts)

    # Set Grid Lines
    # ax.grid(grid, which='both', axis=axis)
    # setGrid(ax, grid, axis=axis)
    setGrid(ax, grid, axis='both')

    if(axis == 'x'):
        ax.xaxis.label.set_color(c)
        offt = ax.get_xaxis().get_offset_text()

        if(side is None):
            if(pos is None):
                side = 'bottom'
            else:
                if(pos < 0.5): side = 'bottom'
                else:            side = 'top'

        if(pos is not None):
            offt.set_y(pos)
            ax.xaxis.set_label_position(side)
            ax.xaxis.set_ticks_position(side)

        if(lim is not None):
            if(np.size(lim) > 2): lim = zmath.minmax(lim)
            ax.set_xlim(lim)

        if(invert): ax.invert_xaxis()
        if(not ticks):
            for tlab in ax.xaxis.get_ticklabels(): tlab.set_visible(False)

    else:
        ax.yaxis.label.set_color(c)
        offt = ax.get_yaxis().get_offset_text()

        if(side is None):
            if(pos is None):
                side = 'left'
            else:
                if(pos < 0.5): side = 'left'
                else:            side = 'right'

        if(pos is not None):
            offt.set_x(pos)

        ax.yaxis.set_label_position(side)
        ax.yaxis.set_ticks_position(side)

        if(lim is not None): ax.set_ylim(lim)

        if(invert): ax.invert_yaxis()
        if(not ticks):
            for tlab in ax.yaxis.get_ticklabels(): tlab.set_visible(False)

    # Set Spine colors
    ax.spines[side].set_color(c)
    if(pos is not None):
        ax.set_frame_on(True)
        ax.spines[side].set_position((trans, pos))
        ax.spines[side].set_visible(True)
        ax.patch.set_visible(False)

    # Set Axis Scaling
    if(scale is not None): _setAxis_scale(ax, axis, scale, thresh=thresh)

    # Set Axis Label
    _setAxis_label(ax, axis, label, fs=fs, c=c)

    if(stretch != 1.0):
        if(axis == 'x'): ax = stretchAxes(ax, xs=stretch)
        elif(axis == 'y'): ax = stretchAxes(ax, ys=stretch)

    offt.set_color(c)
    return ax


def twinAxis(ax, axis='x', pos=1.0, **kwargs):
    """
    """
    if(axis == 'x'):
        tw = ax.twinx()
        setax = 'y'
    elif(axis == 'y'):
        tw = ax.twiny()
        setax = 'x'
    else:
        raise RuntimeError("``axis`` must be either {`x` or `y`}!")

    tw = setAxis(tw, axis=setax, pos=pos, **kwargs)
    return tw


def setLim(ax, axis='y', lo=None, hi=None, data=None, range=False, at='exactly', invert=False):
    """Set the limits (range) of the given, target axis.

    When only ``lo`` or only ``hi`` is specified, the default behavior is to only set that axis
    limit and leave the other bound to its existing value.  When ``range`` is set to `True`, then
    the given axis boumds (``lo``/``hi``) are used as multipliers, i.e.

        >>> Plotting.setLim(ax, lo=0.1, range=True, at='exactly')
        will set the lower bound to be `0.1` times the existing upper bound

    The ``at`` keyword determines whether the given bounds are treated as limits to the bounds,
    or as fixed ('exact') values, i.e.

        >>> Plotting.setLim(ax, lo=0.1, range=True, at='most')
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

    if(axis == 'y'):
        get_lim = ax.get_ylim
        set_lim = ax.set_ylim
    elif(axis == 'x'):
        get_lim = ax.get_xlim
        set_lim = ax.set_xlim
    else:
        raise RuntimeError("``axis`` must be either 'x' or 'y'!")

    lims = np.array(get_lim())

    # Set Range/Span of Limits
    if(range):
        if(lo is not None):
            if(at == AT_EXACTLY):
                lims[0] = lims[1]/lo
            elif(at == AT_LEAST):
                lims[0] = np.max([lims[0], lims[0]/lo])
            elif(at == AT_MOST):
                lims[0] = np.min([lims[0], lims[0]/lo])
        elif(hi is not None):
            if(at == AT_EXACTLY):
                lims[1] = lims[1]*hi
            elif(at == AT_LEAST):
                lims[1] = np.max([lims[1], lims[1]*hi])
            elif(at == AT_MOST):
                lims[1] = np.min([lims[1], lims[1]*hi])
        else:
            raise RuntimeError("``lo`` or ``hi`` must be provided!")

    # Set Limits explicitly
    else:
        if(lo is not None):
            if(at == AT_EXACTLY):
                lims[0] = lo
            elif(at == AT_LEAST):
                lims[0] = np.max([lims[0], lo])
            elif(at == AT_MOST):
                lims[0] = np.min([lims[0], lo])
            else:
                raise ValueError("Unrecognized `at` = '%s'" % (at))
        elif(data is not None):
            lims[0] = np.min(data)

        if(hi is not None):
            if(at == AT_EXACTLY):
                lims[1] = hi
            elif(at == AT_LEAST):
                lims[1] = np.max([lims[1], hi])
            elif(at == AT_MOST):
                lims[1] = np.min([lims[1], hi])
            else:
                raise ValueError("Unrecognized `at` = '%s'" % (at))
        elif(data is not None):
            lims[1] = np.max(data)

    # Actually set the axes limits
    set_lim(lims)
    if(invert):
        if(axis == 'x'): ax.invert_xaxis()
        else:            ax.invert_yaxis()

    return


def zoom(ax, loc, axis='x', scale=2.0):
    """Zoom-in at a certain location on the given axes.
    """

    # Choose functions based on target axis
    if(axis == 'x'):
        axScale = ax.get_xscale()
        lim = ax.get_xlim()
        set_lim = ax.set_xlim
    elif(axis == 'y'):
        axScale = ax.get_yscale()
        lim = ax.get_ylim()
        set_lim = ax.set_ylim
    else:
        raise ValueError("Unrecognized ``axis`` = '%s'!!" % (str(axis)))

    lim = np.array(lim)

    # Determine axis scaling
    if(axScale.startswith('lin')):
        log = False
    elif(axScale.startswith('log')):
        log = True
    else:
        raise ValueError("``axScale`` '%s' not implemented!" % (str(axScale)))

    # Convert to log if appropriate
    if(log):
        lim = np.log10(lim)
        loc = np.log10(loc)

    # Find new axis bounds
    delta = np.diff(zmath.minmax(lim))[0]
    lim = np.array([loc - (0.5/scale)*delta, loc + (0.5/scale)*delta])
    # Convert back to linear if appropriate
    if(log): lim = np.power(10.0, lim)
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

    if(xlog): xlims = np.log10(xlims)
    if(ylog): ylims = np.log10(ylims)

    xlims = [xlims[0] + 0.5*(1.0-xs)*(xlims[1]-xlims[0]),
             xlims[1] + 0.5*(1.0-xs)*(xlims[0]-xlims[1])]

    ylims = [ylims[0] + 0.5*(1.0-ys)*(ylims[1]-ylims[0]),
             ylims[1] + 0.5*(1.0-ys)*(ylims[0]-ylims[1])]

    if(xlog): xlims = np.power(10.0, xlims)
    if(ylog): ylims = np.power(10.0, ylims)

    ax.set_xlim(xlims)
    ax.set_ylim(ylims)

    return ax


def text(fig, pstr, x=0.5, y=0.98, halign='center', valign='top', fs=16, trans=None, **kwargs):
    """Add text to figure.

    Wrapper for the `matplotlib.figure.Figure.text` method.

    Arguments
    ---------
    fig : `matplotlib.figure.Figure` object,
    pstr : str,
        String to be printed.
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
    kwargs : any,
        Additional named arguments passed to `matplotlib.figure.Figure.text`.
        For example, ``color='blue'``, or ``rotation=90``.

    Returns
    -------
    txt : ``matplotlib.text.Text`` object,
        Handle storing the drawn text.

    """
    if trans is None: trans = fig.transFigure
    if valign == 'upper':
        warnings.warn("Use `'top'` not `'upper'`!")
        valign = 'top'

    if valign == 'lower':
        warnings.warn("Use `'bottom'` not `'lower'`!")
        valign = 'bottom'

    txt = fig.text(x, y, pstr, size=fs, family='monospace', transform=trans,
                   horizontalalignment=halign, verticalalignment=valign, **kwargs)

    return txt


def legend(fig, keys, names, x=0.99, y=0.5, halign='right', valign='center', fs=16, trans=None,
           **kwargs):
    """Add a legend to the given figure.

    Wrapper for the `matplotlib.pyplot.Legend` method.

    Arguments
    ---------
    fig : `matplotlib.figure.Figure` object,
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
    kwargs : any,
        Additional named arguments passed to `matplotlib.pyplot.legend`.
        For example, ``ncol=1`` or ``title='Legend Title'``.

    Returns
    -------
    leg : ``matplotlib.legend.Legend`` object,
        Handle storing the drawn legend.

    """
    ax = fig.axes[0]

    if(trans is None): trans = fig.transFigure
    if(valign == 'top'):
        valign = 'upper'

    if(valign == 'bottom'):
        valign = 'lower'

    alignStr = valign + " " + halign

    leg = ax.legend(keys, names, prop={'size': fs, 'family': 'monospace'},
                    loc=alignStr, bbox_transform=trans, bbox_to_anchor=(x, y), **kwargs)

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


def color_cycle(num, ax=None, cmap=plt.cm.spectral, left=0.1, right=0.9):
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

    Returns
    -------
    cols : (`num`,) array_like of RGBA color tuples
        Colors forming the color cycle.

    """
    cmap = _get_cmap(cmap)
    cols = [cmap(it) for it in np.linspace(left, right, num)]
    if ax is not None: ax.set_color_cycle(cols[::-1])
    return cols


def colormap(args, cmap='jet', scale=None):
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

    Returns
    -------
   smap : ``matplotlib.cm.ScalarMappable``
       Scalar mappable object which contains the members:
       `norm`, `cmap`, and the function `to_rgba`.

    """

    if isinstance(cmap, str): cmap = plt.get_cmap(cmap)

    if scale is None:
        if np.size(args) > 1 and np.all(args > 0.0):
            scale = 'log'
        else:
            scale = 'lin'

    log = _scale_to_log_flag(scale)

    # Determine minimum and maximum
    if np.size(args) > 1: min, max = zmath.minmax(args, nonzero=log, positive=log)
    else:                 min, max = 0, np.int(args)-1

    # Create normalization
    if log: norm = mpl.colors.LogNorm(vmin=min, vmax=max)
    else:   norm = mpl.colors.Normalize(vmin=min, vmax=max)

    # Create scalar-mappable
    smap = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    # Bug-Fix something something
    smap._A = []

    return smap


def color_set(num, black=False):
    """Retrieve a (small) set of color-strings with hand picked values.

    Arguments
    ---------
    num : int
        Number of colors to retrieve.
    black : bool
        Include 'black' as the first color.

    Returns
    -------
    cols : (`num`) list of str
        List of `matplotlib` compatible color-strings.

    """
    ncol = len(_COLOR_SET)
    # Make sure enough colors are available
    if (black and num > ncol) or (not black and num > ncol-1):
        errStr = "Only have {} colors {}, `num` ({}) is too large."
        if black: errStr = errStr.format(ncol, "with black", num)
        else:     errStr = errStr.format(ncol-1, "without black", num)
        raise ValueError(errStr)

    if black:
        cols = _COLOR_SET[:num]
    else:
        cols = _COLOR_SET[1:num+1]

    return cols


def setGrid(ax, val, axis='both', below=True):
    """Configure the axes' grid.
    """
    ax.grid(False, which='both', axis='both')
    ax.set_axisbelow(below)
    if val:
        ax.grid(True, which='major', axis=axis, c='0.50', ls='-')
        ax.grid(True, which='minor', axis=axis, c='0.75', ls='-')
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

    # Save as multipage PDF
    if fname.endswith('pdf') and np.size(fig) > 1:
        from matplotlib.backends.backend_pdf import PdfPages
        with PdfPages(fname) as pdf:
            for ff in fig:
                pdf.savefig(figure=ff, **kwargs)
                if close: plt.close(ff)

    else:
        # Save each figure to a different file
        for ii, ff in enumerate(fig):
            # On subsequent figures, append the number to the filename
            if ii == 0:
                usefname = str(fname)
            else:
                usefname = zio.modifyFilename(fname, append='_%d' % (ii))

            ff.savefig(usefname, **kwargs)
            if close: plt.close(ff)

    printStr = "Saved figure to '%s'" % (fname)
    if log is not None: log.log(level, printStr)
    elif verbose: print(printStr)
    return


def strSciNot(val, precman=1, precexp=0):
    """Convert a scalar into a string with scientific notation (latex formatted).

    Arguments
    ---------
        val <flt> : numerical value to convert
        precman <int> : precision of the mantissa (decimal points)
        precexp <int> : precision of the exponent (decimal points)

    Returns
    -------
        str <str> : scientific notation string using latex formatting.

    """
    man, exp = zmath.frexp10(val)
    notStr = "${0:.{2:d}f} \\times \, 10^{{ {1:.{3:d}f} }}$"
    notStr = notStr.format(man, exp, precman, precexp)
    return notStr


def plotHistLine(ax, edges, hist, yerr=None, nonzero=False, positive=False, extend=None,
                 fill=None, **kwargs):
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
    if(kwargs.get('color') is not None): col = kwargs.get('color')
    elif(kwargs.get('c') is not None): col = kwargs.get('c')

    # Extend bin edges if needed
    if(len(edges) != len(hist)+1):
        if(extend == 'left'): edges = np.concatenate([[zmath.extend(edges)[0]], edges])
        elif(extend == 'right'): edges = np.concatenate([edges, [zmath.extend(edges)[1]]])
        else: raise RuntimeError("``edges`` must be longer than ``hist``, or ``extend`` given")

    # Construct plot points to manually create a step-plot
    xval, yval = _histLine(edges, hist)

    # Select nonzero values
    if(nonzero):
        xval = np.ma.masked_where(yval == 0.0, xval)
        yval = np.ma.masked_where(yval == 0.0, yval)

    # Select positive values
    if(positive):
        xval = np.ma.masked_where(yval < 0.0, xval)
        yval = np.ma.masked_where(yval < 0.0, yval)

    # Plot Histogram
    line, = ax.plot(xval, yval, **kwargs)

    # Add yerror-bars
    if(yerr is not None):
        xmid = zmath.midpoints(edges)

        if(nonzero):
            inds = np.where(hist != 0.0)
            ax.errorbar(xmid[inds], hist[inds], yerr=yerr[inds], fmt=None, ecolor=col)
        else:
            ax.errorbar(xmid,       hist,       yerr=yerr,       fmt=None, ecolor=col)

    # Add a fill region
    if(fill is not None):
        ylim = ax.get_ylim()
        if(type(fill) == dict): filldict = fill
        else:                     filldict = dict()

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

    if(zz is None): zz = np.linspace(norm.vmin, norm.vmax, num=len(xx))
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

    if(size is None): size = [30, 6]
    if(color is None): color = ['0.25', COL]
    if(alpha is None): alpha = [0.5, 0.8]

    ax.scatter(xx, yy, s=size[0], color=color[0], alpha=alpha[0], **kwargs)
    pnts = ax.scatter(xx, yy, s=size[1], color=color[1], alpha=alpha[1], **kwargs)

    # Add Contours
    if(cont):
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

    if(scaley.startswith('log')): logy = True
    else:                         logy = False

    # Add Confidence intervals
    if(conf):
        med, cints = zmath.confidenceIntervals(xx, ci=CONF_INTS)
        ax.axvline(med, color='red', ls='--', lw=LW_CONF, zorder=101)
        # Add average
        ave = np.average(xx)
        ax.axvline(ave, color='red', ls=':', lw=LW_CONF, zorder=101)
        for ii, (vals, col) in enumerate(zip(cints, CONF_COLS)):
            ax.axvspan(*vals, color=col, alpha=CONF_ALPHA)

    # Create a given number of log-spaced bins
    #     If not log-spaced, then `ax.hist` will do the same
    if(isinstance(bins, numbers.Integral) and scalex.startswith('log')):
        bins = zmath.spacing(xx, num=bins, )

    cnts, bins, bars = ax.hist(xx, bins, histtype='bar', log=logy, alpha=HIST_ALPHA,
                               rwidth=0.8, color=COL_CORR, zorder=100, **kwargs)

    # Dont let lower y-lim be less than 0.8 with log-scaling
    if(scaley.startswith('log')):
        # setLim(ax, 'y', lo=0.8, at='least')   <=== This isn't working for some reason!  FIX
        ylim = np.array(ax.get_ylim())
        if(ylim[0] < 0.8):
            ylim[0] = 0.8
            ax.set_ylim(ylim)

    return bars


def plotConfFill(ax, rads, med, conf, color='red', fillalpha=0.5, lw=1.0,
                 filter=None, outline='0.5', **kwargs):
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

    ll, = ax.plot(rads, med, '-', color=color, lw=lw)

    # `conf` has shape ``(num-rads, num-conf-ints, 2)``
    if(conf.ndim == 2):
        conf = conf.reshape(len(rads), 1, 2)
    elif(conf.ndim != 3):
        raise ValueError("`conf` must be 2 or 3 dimensions!")

    if filter is not None:
        filter = zmath._comparisonFunction(filter)

    numConf = np.shape(conf)[-2]
    confPatches = []
    # Iterate over confidence intervals
    for jj in xrange(numConf):
        # Set fill-opacity
        if(np.size(fillalpha) == numConf): falph = fillalpha
        else:                              falph = np.power(fillalpha, jj+1)

        # Create a dummy-patch to return for a legend of confidence-intervals
        pp = ax.fill(np.nan, np.nan, color=color, alpha=falph)
        confPatches.append(pp[0])

        xx = np.array(rads)
        ylo = np.array(conf[:, jj, 0])
        yhi = np.array(conf[:, jj, 1])
        if filter:
            inds = np.where(filter(ylo, 0.0) & filter(yhi, 0.0))
            xx = xx[inds]
            ylo = ylo[inds]
            yhi = yhi[inds]

        # Fill between confidence intervals
        ax.fill_between(xx, ylo, yhi, alpha=falph, color=color, **kwargs)

        # Create dummy-patch for the median-line and fill-color, for a legend
        if(jj == 0):
            pp = ax.fill(np.nan, np.nan, color=color, alpha=falph, **kwargs)
            # Create overlay of lines and patches
            linePatch = (pp[0], ll)

    # Plot Median Line
    #    Plot black outline to improve contrast
    if outline is not None:
        ax.plot(rads, med, '-', color=outline, lw=2*lw, alpha=_LW_OUTLINE)
    ll, = ax.plot(rads, med, '-', color=color, lw=lw)

    return linePatch, confPatches


def _setAxis_scale(ax, axis, scale, thresh=None):
    if(scale.startswith('lin')): scale = 'linear'
    if(scale == 'symlog'): thresh = 1.0
    if(axis == 'x'): ax.set_xscale(scale, linthreshx=thresh)
    elif(axis == 'y'): ax.set_yscale(scale, linthreshy=thresh)
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
    if scale.startswith('log'): log = True
    elif scale.startswith('lin'): log = False
    else: raise ValueError("Unrecognized `scale` '%s'; must start with 'log' or 'lin'." % (scale))
    return log


def _clean_scale(scale):
    """Cleanup a 'scaling' string to be matplotlib compatible.
    """
    scale = scale.lower()
    if scale.startswith('lin'): scale = 'linear'
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

'''
def rescale(ax, which='both'):
    """
    """

    if(which == 'x'):
        scaley = False
        scalex = True
    elif(which == 'y'):
        scaley = True
        scalex = True
    elif(which == 'both'):
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
