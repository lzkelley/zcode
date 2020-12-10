"""
"""
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import six

from zcode.plot import _PAD


__all__ = ["backdrop", "extent", "full_extent", "position_to_extent", "rect_for_inset",
           "transform", "zoom_effect"]


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
        if draw:
            fig.patches.append(rect)
        pats.append(rect)

    if len(pats) == 1:
        return pats[0]
    return pats


def extent(ax, pad=0.0, invert=None, fig=None):
    """Get the full extent of an axes, including axes labels, tick labels, and titles.

    From: 'stackoverflow.com/questions/14712665/'
    """
    # Draw text objects so extents are defined
    ax.figure.canvas.draw()
    bbox = ax.get_window_extent().expanded(1.0 + pad, 1.0 + pad)
    if invert:
        bbox = bbox.transformed(invert.inverted())

    return bbox


def full_extent(ax, pad=0.0, invert=None):
    """Get the full extent of an axes, including axes labels, tick labels, and titles.

    From: 'stackoverflow.com/questions/14712665/'
    """
    # Draw text objects so extents are defined
    ax.figure.canvas.draw()
    rend = ax.figure.canvas.get_renderer()
    if isinstance(ax, mpl.axes.Axes):
        items = ax.get_xticklabels() + ax.get_yticklabels()
        items += [ax, ax.title, ax.xaxis.label, ax.yaxis.label]
        # items += [ax, ax.title]
        use_items = []
        for item in items:
            if isinstance(item, mpl.text.Text) and len(item.get_text()) == 0:
                continue
            use_items.append(item)
        # bbox = mpl.transforms.Bbox.union([item.get_window_extent() for item in use_items])
        bbox = _set_extents(use_items, union=True, pad=0.0, renderer=rend)
    elif isinstance(ax, mpl.legend.Legend):
        bbox = ax.get_frame().get_bbox()
    elif isinstance(ax, mpl.text.Text):
        bbox = ax.get_tightbbox(plt.gcf().canvas.get_renderer())
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
    # try:
    ax_bbox = ref.get_position()
    # except AttributeError:
    #     ax_bbox = ref.get_bbox_to_anchor()
    #     print("old_loc = ", ax_bbox)

    if halign.startswith('r'):
        dx = ax_bbox.x1 - bbox.x1
    elif halign.startswith('l'):
        dx = ax_bbox.x0 - bbox.x0
    elif halign.startswith('c'):
        dx = 0.5*(ax_bbox.x0 + ax_bbox.x1) - 0.5*(bbox.x0 + bbox.x1)
    else:
        raise ValueError("`halign` = '{}' must start with 'l' or 'r'.".format(halign))

    if valign.startswith('l'):
        dy = ax_bbox.y0 - bbox.y0
    elif valign.startswith('u'):
        dy = ax_bbox.y1 - bbox.y1
    elif valign.startswith('c'):
        dy = 0.5*(ax_bbox.y0 + ax_bbox.y1) - 0.5*(bbox.y0 + bbox.y1)
    else:
        raise ValueError("`valign` = '{}' must start with 'l' or 'u'.".format(valign))

    if len(loc) == 2:
        new_loc = [loc[0]+dx, loc[1]+dy, ax_bbox.width, ax_bbox.height]
    elif len(loc) == 4:
        new_loc = [loc[0]+dx, loc[1]+dy, loc[2], loc[3]]
    else:
        raise ValueError("`loc` must be 2 or 4 long: [x, y, (width, height)]")

    # try:
    item.set_position(new_loc)
    # except AttributeError:
    #     print("new_loc = ", new_loc)
    #     item.set_bbox_to_anchor(new_loc)

    return


def rect_for_inset(parent, loc='tl', width=None, height=None,
                   width_frac=0.25, height_frac=0.25, pad=None):
    """Construct the rectangle to describe an inset axes relative to the parent object.
    """
    pos = parent.get_position()
    wid = pos.width * width_frac if (width is None) else width
    hit = pos.height * height_frac if (height is None) else height

    if pad is None:
        pad = [_PAD, _PAD]

    if np.isscalar(pad):
        pad = [pad, pad]

    if loc[0] == 'b':
        yy = pos.y0 + pad[1]
    elif loc[0] == 't':
        yy = pos.y1 - pad[1] - hit
    else:
        raise ValueError("`loc[0]` must be ['b', 't']!")

    if loc[1] == 'l':
        xx = pos.x0 + pad[0]
    elif loc[1] == 'r':
        xx = pos.x1 - pad[0] - wid
    else:
        raise ValueError("`loc[1]` must be ['l', 'r']!")

    rect = [xx, yy, wid, hit]
    return rect


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


def zoom_effect(ax_main, ax_zoom, lines=[['tl', 'tl'], ['bl', 'bl']], **kwargs):
    """Add connection lines between two axes creating a zoom effect.

    Arguments
    ---------
    ax_main : the base/main axes instance
    ax_zoom : the zoomed-in axes instance
    lines : (2, 2), str or int
        Specification for which corners to connect for the two lines,
        The format is [[line1-axis1, line1-axis2], [line2-axis1, line2-axis2]]
        Each element should be one of ['tl', 'tr', 'bl', 'br']

    """

    tt = ax_zoom.transScale + (ax_zoom.transLimits + ax_main.transAxes)
    # This zooms into just the x-axis (and uses full y-axis range):
    # trans = mpl.transforms.blended_transform_factory(ax_main.transData, tt)
    # This zooms into both x and y axes:
    trans = ax_main.transData

    bbox_zoom = ax_zoom.bbox
    bbox = mpl.transforms.TransformedBbox(ax_zoom.viewLim, trans)

    kwargs.setdefault('color', 'red')
    kwargs.setdefault('lw', 0.5)
    kwargs.setdefault('alpha', 0.25)
    prop_patches = kwargs.copy()
    prop_patches["ec"] = kwargs['color']
    prop_patches["fc"] = "none"
    prop_patches["lw"] = 0.5
    prop_patches["alpha"] = 0.2

    l1a1, l1a2 = lines[0]
    l2a1, l2a2 = lines[1]
    
    c1, c2, bbox_patch1, bbox_patch2, p = _connect_bbox(
        bbox_zoom, bbox, l1a1, l1a2, l2a1, l2a2, 
        prop_lines=kwargs, prop_patches=prop_patches)

    # ax_zoom.add_patch(bbox_patch1)
    ax_main.add_patch(bbox_patch2)
    ax_main.add_patch(c1)
    ax_main.add_patch(c2)
    # ax_main.add_patch(p)

    return c1, c2, bbox_patch1, bbox_patch2, p


def _connect_bbox(bbox1, bbox2, l1a1, l1a2, l2a1, l2a2, prop_lines, prop_patches=None):
    """Add connectors between two bbox's.

    l1a1 : line-1 axis-1  (connects from here to l1a2)
    l1a2 : line-1 axis-2
    l2a1 : line-2 axis-1  (connects from here to l2a2)
    l2a2 : line-2 axis-2

    """

    def _corner_trans(arg):
        if isinstance(arg, int):
            return arg
        if not isinstance(arg, str):
            err = (
                f"This function translates from a 'str' specification to an 'int'; "
                f"recieved {type(arg)}!"
            )
            raise ValueError(err)
        arg = arg.lower()
        if (arg == 'll') or (arg == 'bl'):
            return 3
        elif (arg == 'lr') or (arg == 'br'):
            return 4
        elif (arg == 'tl') or (arg == 'ul'):
            return 2
        elif (arg == 'tr') or (arg == 'ur'):
            return 1
        else:
            raise ValueError(f"Unrecognized position specification '{arg}'!")
        
    from mpl_toolkits.axes_grid1.inset_locator import BboxPatch, BboxConnector, BboxConnectorPatch
    
    if prop_patches is None:
        prop_patches = prop_lines.copy()
        prop_patches["alpha"] = prop_patches.get("alpha", 1)*0.2

    # Convert from str specifications to integers as needed
    l1a1, l1a2, l2a1, l2a2 = [_corner_trans(aa) for aa in [l1a1, l1a2, l2a1, l2a2]]
        
    c1 = BboxConnector(bbox2, bbox1, loc1=l1a1, loc2=l1a2, **prop_lines)
    c1.set_clip_on(False)
    c2 = BboxConnector(bbox2, bbox1, loc1=l2a1, loc2=l2a2, **prop_lines)
    c2.set_clip_on(False)

    bbox_patch1 = BboxPatch(bbox1, **prop_patches)
    bbox_patch2 = BboxPatch(bbox2, **prop_patches)

    p = BboxConnectorPatch(
        bbox1, bbox2, loc1a=l1a1, loc2a=l1a2, loc1b=l2a1, loc2b=l2a2, **prop_patches)
    p.set_clip_on(False)

    return c1, c2, bbox_patch1, bbox_patch2, p


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

    pad = np.atleast_1d(pad)
    if pad.size == 1:
        pad = np.concatenate([pad, pad])

    if loc[0] == 't' or loc[0] == 'u':
        if valign is None:
            valign = 'top'
        if y is None:
            y = 1 - pad[1]
    elif loc[0] == 'b' or loc[0] == 'l':
        if valign is None:
            valign = 'bottom'
        if y is None:
            y = pad[1]
    elif loc[0] == 'c':
        if valign is None:
            valign = 'center'
        if y is None:
            y = 0.5

    if loc[1] == 'l':
        if halign is None:
            halign = 'left'
        if x is None:
            x = pad[0]
    elif loc[1] == 'r':
        if halign is None:
            halign = 'right'
        if x is None:
            x = 1 - pad[0]
    elif loc[1] == 'c':
        if halign is None:
            halign = 'center'
        if x is None:
            x = 0.5

    return x, y, halign, valign


def _extents(ax, pad=0.0, invert=None, group=False):
    """Get the extents of an axes, including axes labels, tick labels, and titles.

    Adapted From: 'stackoverflow.com/questions/14712665/'
    """
    # For text objects, we need to draw the figure first, otherwise the extents
    # are undefined.
    ax.figure.canvas.draw()
    rend = ax.figure.canvas.get_renderer()
    kw = dict(pad=pad, renderer=rend)
    if group:
        items = [ax, ax.title, ax.xaxis.label, ax.yaxis.label]
        bboxes = [_set_extents(ax.get_xticklabels(), union=True, **kw)]
        bboxes += [_set_extents(ax.get_yticklabels(), union=True, **kw)]
        bboxes += _set_extents(items, union=False, **kw)
    else:
        items = ax.get_xticklabels() + ax.get_yticklabels()
        items += [ax, ax.title, ax.xaxis.label, ax.yaxis.label]
        # bboxes = [it.get_window_extent().expanded(1.0 + pad, 1.0 + pad) for it in items]
        bboxes = _set_extents(items, union=False, **kw)

    if invert:
        bboxes = [bb.transformed(invert.inverted()) for bb in bboxes]

    return bboxes


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


def _set_extents(items, pad=0.0, union=False, **kw):
    bboxes = [it.get_window_extent(**kw).expanded(1.0 + pad, 1.0 + pad) for it in items]
    if union:
        bboxes = mpl.transforms.Bbox.union(bboxes)
    return bboxes
