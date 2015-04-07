"""
General plotting functions.

Functions
---------
 - unifyAxesLimits() : given a list of axes, set all limits to match flobal extrema
 - setColorCycle()   : create a cycle of the given number of colors
 - setAxis()         : function to set many different axis properties at once
 - twinAxis()        : easily create and set a new twin axis (like `twinx()` or `twiny()`)
 - histPlotLine()


"""


import numpy as np
import astropy   as ap
import scipy as sp
import matplotlib as mpl
from matplotlib import pyplot as plt
from datetime import datetime

CMAP = plt.cm.gist_heat

FS = 10

VALID_SIDES = [ None, 'left', 'right', 'top', 'bottom' ]



def watermark(fig, run, fs=FS):
    wm_str = "Illustris-%d\n%s" % (run, datetime.now().ctime())
    text = fig.text(0.98, 0.02, wm_str, horizontalalignment='right', fontsize=FS,
                    verticalalignment='bottom', transform=plt.gcf().transFigure)
    return text



def addParameterString(fig, str, xx=0.98, yy=0.1, halign='right', valign='bottom', fs=16):
    txt = fig.text(xx, yy, str, size=fs, family='monospace',
                   horizontalalignment=halign, verticalalignment=valign,
                   transform=fig.transFigure)

    return txt


def unifyAxesLimits(axes, axis='y'):
    """
    Given a list of axes, set all limits to match global extrema.
    """

    assert axis in ['x','y'], "``axis`` must be either 'x' or 'y' !!"

    if( axis == 'y' ):
        lims = np.array([ax.get_ylim() for ax in axes])
    else:
        lims = np.array([ax.get_xlim() for ax in axes])

    lo = np.min(lims[:,0])
    hi = np.max(lims[:,1])

    for ax in axes:
        if( axis == 'y' ):
            ax.set_ylim([lo,hi])
        else:
            ax.set_xlim([lo,hi])

    return np.array([lo,hi])



def setColorCycle(num, ax=None, cmap=plt.cm.spectral, left=0.1, right=0.9):
    if(ax == None): ax = plt.gca()
    cols = [cmap(it) for it in np.linspace(left, right, num)]
    ax.set_color_cycle(cols[::-1])
    return cols


def plotRect(ax, loc):
    rect = mpl.patches.Rectangle((loc[0], loc[1]), loc[2], loc[3],
                                 alpha=0.4, facecolor='None', ls='dashed', lw=1.0, transform=ax.transData)
    ax.add_patch(rect)
    return


def twinAxis(ax, twin='x', fs=12, c='black', pos=1.0, trans='axes', label=None, scale=None, thresh=None, ts=None, side=None, lim=None, grid=True):
    assert twin in ['x','y'], "``twin`` must be either `x` or `y`!"
    assert trans in ['axes','figure'], "``trans`` must be either `axes` or `figure`!"

    if( scale == 'symlog' and thresh is None ):
        thresh = 1.0

    if( twin == 'x' ):
        tw = ax.twinx()
        '''
        if( side is None ): 
            if( pos > 0.0 ): side = 'right'
            else:            side = 'left'
        '''
        tw = setAxis(tw, axis='y', fs=fs, c=c, pos=pos, trans=trans, label=label, scale=scale, thresh=thresh, side=side, ts=ts, grid=grid, lim=lim)
    else:
        tw = ax.twiny()
        #if( side is None ): side = 'top'
        tw = setAxis(tw, axis='x', fs=fs, c=c, pos=pos, trans=trans, label=label, scale=scale, thresh=thresh, side=side, ts=ts, grid=grid, lim=lim)

    return tw


def _setAxis_scale(ax, axis, scale, thresh=None):
    if(   axis == 'x' ): ax.set_xscale(scale, linthreshx=thresh)
    elif( axis == 'y' ): ax.set_yscale(scale, linthreshy=thresh)
    else: raise RuntimeError("Unrecognized ``axis`` = %s" % (axis))
    return


def _setAxis_label(ax, axis, label, fs=12, c='black'):
    if(   axis == 'x' ): ax.set_xlabel(label, size=fs, color=c)
    elif( axis == 'y' ): ax.set_ylabel(label, size=fs, color=c)
    else: raise RuntimeError("Unrecognized ``axis`` = %s" % (axis))
    return



def setAxis(ax, axis='x', fs=12, c='black', pos=None, trans='axes', label=None, scale=None, thresh=None, side=None, ts=8, grid=True, lim=None):
    assert axis  in ['x','y'],                          "``axis`` must be either `x` or `y`!"
    assert trans in ['axes','figure'],                  "``trans`` must be either `axes` or `figure`!"
    assert side in VALID_SIDES, "``side`` must be in '%s'" % (VALID_SIDES)

    # Set tick colors
    ax.tick_params(axis=axis, which='both', colors=c, labelsize=fs)
    ax.tick_params(axis=axis, which='major', size=ts)

    # Set Grid Lines
    ax.grid(grid, axis=axis)

    if( axis == 'x' ):
        ax.xaxis.label.set_color(c)
        offt = ax.get_xaxis().get_offset_text()

        if( side is None ):
            if( pos is None ):
                side = 'bottom'
            else:
                if( pos < 0.5 ): side = 'bottom'
                else:            side = 'top'

        if( pos is not None ):
            offt.set_y(pos)
            ax.xaxis.set_label_position(side)
            ax.xaxis.set_ticks_position(side)

        if( lim is not None ): ax.set_xlim( lim )

    else:
        ax.yaxis.label.set_color(c)
        offt = ax.get_yaxis().get_offset_text()

        if( side is None ):
            if( pos is None ):
                side = 'left'
            else:
                if( pos < 0.5 ): side = 'left'
                else:            side = 'right'


        if( pos is not None ):
            offt.set_x(pos)
            ax.yaxis.set_label_position(side)
            ax.yaxis.set_ticks_position(side)

            
        if( lim is not None ): ax.set_ylim( lim )



    # Set Spine colors
    ax.spines[side].set_color(c)
    if( pos is not None ):
        ax.set_frame_on(True)
        ax.spines[side].set_position((trans, pos))
        ax.spines[side].set_visible(True)
        ax.patch.set_visible(False)

    # Set Axis Scaling
    if( scale is not None ):
        VALID_SCALES = ['linear', 'log', 'symlog']
        assert scale in VALID_SCALES,   "``scale`` must be in '%s'!" % (VALID_SCALES)
        _setAxis_scale(ax, axis, scale, thresh=thresh)

    # Set Axis Label
    if( label is not None ):
        _setAxis_label(ax, axis, label, fs=fs, c=c)


    offt.set_color(c)

    return ax



def histPlotLine(values, bins, ax=None, weights=None, ls='-', lw=1.0, color='k', ave=False, scale=None, label=None):
    """
    Manually plot a histogram.

    Uses numpy.histogram to obtain binned values, then plots them manually
    as connected lines with the given parameters.  If `weights` are provided,
    they are the values summed for each bin instead of 1.0 for each entry in
    `values`.

    Parameters
    ----------
    ax : object, matplotlib.axes
        Axes on which to make plot

    values : array_like, scalar
        Array of values to be binned and plotted.  Each entry which belongs in
        a bin, increments that bin by 1.0 or the corresponding entry in
        `weights` if it is provived.

    bins : array_like, scalar
        Edges of bins to use for histogram, with both left and right edges.
        If `bins` has length N, there will be N-1 bins.

    weights : array_like, scalar, optional
        Array of the same length as `values` which contains the weights to be
        added to each bin.

    ls : str, optional
        linestyle to plot with

    lw : scalar, optional
        lineweight to plot with

    color : str, optional
        color of line to plot with

    scale : scalar or array of scalars
        Rescale the resulting histogram by this/these values
        (e.g. 1/binVol will convert to density)

    label : str, optional
        label to associate with plotted histogram line

    Returns
    -------
    ll : object, matplotlib.lines.Line2D
        Line object which was plotted to axes `ax`
        (can then be used to make a legend, etc)

    hist : array, scalar
        The histogram which is plotted

    """

    hist,edge = np.histogram( values, bins=bins, weights=weights )

    # Find the average of each weighed bin instead.
    if( ave and weights is not None ):
        hist = [ hh/nn if nn > 0 else 0.0
                 for hh,nn in zip(hist,np.histogram( values, bins=bins)[0]) ]

    # Rescale the bin values
    if( scale != None ):
        hist *= scale

    yval = np.concatenate([ [hh,hh] for hh in hist ])
    xval = np.concatenate([ [edge[jj],edge[jj+1]] for jj in range(len(edge)-1) ])

    if( ax == None ):
        ll = None
    else:
        ll, = ax.plot( xval, yval, ls, lw=lw, color=color, label=label)

    return ll, hist



def histLine(edges, hist):
    yval = np.concatenate([ [hh,hh] for hh in hist ])
    xval = np.concatenate([ [edges[jj],edges[jj+1]] for jj in range(len(edges)-1) ])
    return xval, yval


def plotHistLine(ax, edges, hist, color='black', label=None, lw=1.0, ls='-', alpha=1.0):
    xval, yval = histLine(edges, hist)
    line, = ax.plot( xval, yval, ls=ls, lw=lw, color=color, label=label, alpha=alpha)
    return line
    





def createFigures(nfigs=1, figsize=None, axsize=None, axpos=None):

    # Set unset parameters to defaults
    if( figsize == None ): figsize = FIG_SIZE
    if( axsize == None ): axsize = AX_SIZE
    if( axpos == None ): axpos = AX_POS

    # Make parameters iterable
    if( not np.iterable(axsize) ): axsize = [ axsize ]
    if( not np.iterable(axpos) ): axpos = [ axpos ]

    # Create figures, add axes
    figs = [ plt.figure(figsize=figsize) for ii in range(nfigs) ]
    for ff in figs:
        for axp,axs in zip(axpos, axsize):
            ff.add_axes(axp + axs)

    return figs


'''
def createFigure(left=AX_LEFT, right=AX_RIGHT, bottom=AX_BOTTOM, top=AX_TOP):

    fig = plt.figure(figsize=FIG_SIZE)
    pos = [left, bottom]
    size = [1.0-left-right, 1.0-bottom-top]
    fig.add_axes(pos + size)

    return fig
'''

def saveFigure(fname, fig, verbose=True):
    fig.savefig(fname)
    if( verbose ): print "Saved figure to '%s'" % (fname)
    return



def clear_frame(ax=None): 
    # Taken from a post by Tony S Yu
    if ax is None: ax = plt.gca() 
    ax.xaxis.set_visible(False) 
    ax.yaxis.set_visible(False) 
    for spine in ax.spines.itervalues(): spine.set_visible(False) 

    return



def make_segments(x, y):
    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb

    Create list of line segments from x and y coordinates, in the correct format for LineCollection:
    an array of the form   [numlines, (points per line), 2 (x and y)] array
    """

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    return segments



def colorline(x, y, z, cmap=plt.cm.jet, norm=plt.Normalize(0.0, 1.0), lw=3, alpha=1.0):
    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb

    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """
    
    z = np.asarray(z)
    segments = make_segments(x, y)
    lc = mpl.collections.LineCollection(segments, array=z, cmap=cmap, norm=norm, linewidth=lw, alpha=alpha)
    
    ax = plt.gca()
    ax.add_collection(lc)
    
    return lc


def cmapColors(args, cmap=plt.cm.jet, scale='log'):
    if( scale == 'log' ): norm = mpl.colors.LogNorm(vmin=np.min(args), vmax=np.max(args))
    else:                 norm = mpl.colors.Normalize(vmin=np.min(args), vmax=np.max(args))

    smap = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    return smap.to_rgba, norm, cmap
